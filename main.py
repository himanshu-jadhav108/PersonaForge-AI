"""
main.py — PersonaForge AI · FastAPI Application

Endpoints:
  GET  /              → serve index.html
  POST /upload        → upload image + video, return session_id
  POST /preview       → generate 3–5 s preview clip (background)
  POST /process       → full processing (can skip preview frames)
  GET  /status/{jid}  → poll job status with stage, progress, message
  GET  /download/{fn} → stream output file
  GET  /jobs          → list recent jobs (history)
"""

import time
import uuid
import logging
import asyncio
import datetime
from pathlib import Path
from typing import Optional
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from video_utils import (
    extract_frames, rebuild_video, resize_video,
    get_video_info, get_file_size_mb, cleanup_temp_dirs,
)
from face_swap import FaceSwapper, FaceSwapError, QualityMode
from models.model_manager import check_models

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("personaforge.main")

# ─── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
FRAMES_DIR  = BASE_DIR / "temp_frames"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_DIR  = BASE_DIR / "static"

for d in [UPLOADS_DIR, FRAMES_DIR, OUTPUTS_DIR, STATIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Job Store (in-memory, FIFO eviction at 100 jobs) ─────────────────────────
JOBS: "OrderedDict[str, dict]" = OrderedDict()
MAX_JOBS = 100

def _new_job(session_id: str, kind: str) -> str:
    """Create a new job entry, evict oldest if over limit."""
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id":              job_id,
        "session_id":      session_id,
        "kind":            kind,          # "preview" | "full"
        "status":          "queued",
        "stage":           "queued",
        "progress":        0,
        "message":         "Job queued",
        "output":          None,
        "file_size_mb":    None,
        "device":          None,
        "similarity_score": None,
        "orientation":     None,
        "input_width":     None,
        "input_height":    None,
        "resize_mode":     "maintain",
        "created_at":      datetime.datetime.utcnow().isoformat() + "Z",
    }
    while len(JOBS) > MAX_JOBS:
        JOBS.popitem(last=False)
    return job_id

def _update(job_id: str, status: str, stage: str, progress: int, message: str, **extra):
    if job_id not in JOBS:
        return
    JOBS[job_id].update(dict(
        status=status, stage=stage, progress=progress, message=message, **extra
    ))
    logger.info("[%s] %d%% [%s] %s", job_id[:8], progress, stage, message)

# ─── Quality → bitrate & target height map ─────────────────────────────────────
_QUALITY_CONFIG = {
    "fast":     {"height": 480, "bitrate": "1M"},
    "balanced": {"height": 720, "bitrate": "3M"},
    "high":     {"height": 0,   "bitrate": "6M"},  # 0 = original resolution
}

# ─── FastAPI App ───────────────────────────────────────────────────────────────
# ─── App Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown logic."""
    try:
        check_models(auto_download=False)
        logger.info("[lifespan] Models validated on startup.")
    except FileNotFoundError as exc:
        logger.error("[lifespan] Model validation failed: %s", exc)
        raise RuntimeError(str(exc)) from exc
    yield
    # No cleanup required for now

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PersonaForge AI",
    description="High-quality AI face swapping with GPU acceleration.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")



ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def _check_ext(filename: str, allowed: set, label: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Invalid {label} format '{ext}'. Allowed: {', '.join(sorted(allowed))}")
    return ext


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "Frontend not found.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/upload", summary="Upload source face image and target video")
async def upload_files(
    image: UploadFile = File(..., description="Source face image"),
    video: UploadFile = File(..., description="Target video"),
):
    _check_ext(image.filename, ALLOWED_IMAGE_EXTS, "image")
    _check_ext(video.filename, ALLOWED_VIDEO_EXTS, "video")

    session_id  = uuid.uuid4().hex
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    img_ext  = Path(image.filename).suffix.lower()
    img_path = session_dir / f"source_face{img_ext}"
    img_data = await image.read()
    if not img_data:
        raise HTTPException(400, "Uploaded image is empty.")
    img_path.write_bytes(img_data)

    vid_ext  = Path(video.filename).suffix.lower()
    vid_path = session_dir / f"target_video{vid_ext}"
    vid_data = await video.read()
    if not vid_data:
        raise HTTPException(400, "Uploaded video is empty.")
    vid_path.write_bytes(vid_data)

    logger.info("Session %s: image %d B, video %d B", session_id, len(img_data), len(vid_data))

    try:
        info = get_video_info(str(vid_path))
    except Exception as e:
        info = {"error": str(e)}

    return JSONResponse({
        "session_id":  session_id,
        "image_path":  str(img_path),
        "video_path":  str(vid_path),
        "video_info":  info,
        "message":     "Upload successful. Use /preview or /process to continue.",
    })


@app.post("/preview", summary="Generate a short preview clip (first 3–5 s)")
async def preview_faceswap(
    background_tasks: BackgroundTasks,
    session_id: str = Query(...),
    quality:    str = Query("balanced", enum=["fast", "balanced", "high"]),
    face_index: int = Query(-1, description="-1=all faces, 0..n=specific face"),
    duration:   float = Query(4.0, description="Preview duration in seconds"),
    resize_mode: str = Query("maintain", enum=["maintain", "crop_portrait"]),
):
    session_dir, img_path, vid_path = _resolve_session(session_id)
    job_id = _new_job(session_id, "preview")
    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id, session_id=session_id,
        img_path=img_path, vid_path=vid_path,
        quality=quality, face_index=face_index,
        preview_seconds=duration,
        resize_mode=resize_mode,
    )
    logger.info("Preview job %s queued for session %s", job_id, session_id)
    return JSONResponse({"job_id": job_id, "message": "Preview started. Poll /status/{job_id}."})


@app.post("/process", summary="Start full face-swap processing")
async def process_faceswap(
    background_tasks: BackgroundTasks,
    session_id: str = Query(...),
    quality:    str = Query("balanced", enum=["fast", "balanced", "high"]),
    face_index: int = Query(-1),
    resize_mode: str = Query("maintain", enum=["maintain", "crop_portrait"]),
):
    session_dir, img_path, vid_path = _resolve_session(session_id)
    job_id = _new_job(session_id, "full")
    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id, session_id=session_id,
        img_path=img_path, vid_path=vid_path,
        quality=quality, face_index=face_index,
        preview_seconds=None,
        resize_mode=resize_mode,
    )
    logger.info("Full job %s queued for session %s", job_id, session_id)
    return JSONResponse({"job_id": job_id, "message": "Processing started. Poll /status/{job_id}."})


@app.get("/status/{job_id}", summary="Poll job progress")
async def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    return JSONResponse(JOBS[job_id])


@app.get("/jobs", summary="List recent job history")
async def list_jobs(limit: int = Query(20, ge=1, le=100)):
    recent = list(reversed(list(JOBS.values())))[:limit]
    return JSONResponse({"jobs": recent, "total": len(JOBS)})


@app.get("/download/{filename}", summary="Download an output video")
async def download_video(filename: str):
    safe_name   = Path(filename).name
    output_path = OUTPUTS_DIR / safe_name
    if not output_path.exists():
        raise HTTPException(404, f"Output file '{safe_name}' not found.")
    return FileResponse(str(output_path), media_type="video/mp4", filename=safe_name)


# ─── Session Resolver ──────────────────────────────────────────────────────────

def _resolve_session(session_id: str) -> tuple[Path, str, str]:
    session_dir = UPLOADS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, f"Session '{session_id}' not found.")
    imgs = list(session_dir.glob("source_face.*"))
    vids = list(session_dir.glob("target_video.*"))
    if not imgs:
        raise HTTPException(400, "Source face image not found for this session.")
    if not vids:
        raise HTTPException(400, "Target video not found for this session.")
    return session_dir, str(imgs[0]), str(vids[0])


# ─── Background Pipeline ───────────────────────────────────────────────────────

async def _run_pipeline(
    job_id:          str,
    session_id:      str,
    img_path:        str,
    vid_path:        str,
    quality:         str  = "balanced",
    face_index:      int  = -1,
    preview_seconds: Optional[float] = None,
    resize_mode:     str = "maintain",
):
    """
    Unified preview + full processing pipeline.

    Preview mode:  preview_seconds=N  → extracts only first N seconds of frames.
    Full mode:     preview_seconds=None → extracts all frames.
    """
    import time
    qcfg    = _QUALITY_CONFIG.get(quality, _QUALITY_CONFIG["balanced"])
    qmode   = QualityMode(quality)
    bitrate = qcfg["bitrate"]
    height  = qcfg["height"]
    is_preview = preview_seconds is not None

    def upd(stage, progress, message, **extra):
        _update(job_id, "running", stage, progress, message, **extra)

    t_total = time.perf_counter()
    try:
        # ── Init swapper ───────────────────────────────────────────────────
        upd("processing", 5, "Loading AI models…")
        swapper = FaceSwapper()
        device  = swapper.get_execution_provider()
        mode    = swapper.get_mode()   # 'gpu' | 'cpu'
        JOBS[job_id]["device"] = device
        JOBS[job_id]["mode"]   = mode
        JOBS[job_id]["resize_mode"] = resize_mode
        upd("processing", 10, f"Models ready on {device}.", device=device)

        # ── CPU override: lock to 480p + 1M regardless of requested quality ───
        if mode == "cpu":
            height  = min(height, 480) if height > 0 else 480
            bitrate = "1M"
            upd("processing", 11,
                "Running in CPU optimized mode (faster settings applied)…")
        else:
            upd("processing", 11, "Running in GPU mode (high quality)…")

        # ── Resize ────────────────────────────────────────────────────────
        upd("processing", 12, "Checking resolution…")
        loop = asyncio.get_event_loop()
        info = get_video_info(vid_path)
        in_w = int(info.get("width", 0) or 0)
        in_h = int(info.get("height", 0) or 0)
        orientation = info.get("orientation", "unknown")
        JOBS[job_id].update({
            "input_width": in_w,
            "input_height": in_h,
            "orientation": orientation,
        })
        resized_path = vid_path
        need_resize = (height > 0 and max(in_w, in_h) > height) or (resize_mode == "crop_portrait")
        if need_resize:
            resized_path = vid_path.replace(Path(vid_path).suffix, f"_{height}p_{resize_mode}.mp4")
            upd("processing", 14, f"Preparing {orientation} video ({resize_mode})…")
            await loop.run_in_executor(None, resize_video, vid_path, resized_path, height, resize_mode)
        else:
            upd("processing", 14, f"Resolution OK ({in_w}x{in_h}, {orientation}).")

        # ── Extract frames ────────────────────────────────────────────────
        suffix  = "_preview" if is_preview else ""
        frames_dir  = str(FRAMES_DIR / f"{job_id}{suffix}")
        upd("processing", 16, f"Extracting frames{' (preview)' if is_preview else ''}…")
        fps, total_frames, audio_path = await loop.run_in_executor(
            None, extract_frames, resized_path, frames_dir, preview_seconds
        )
        upd("processing", 30, f"Extracted {total_frames} frames @ {fps:.1f} fps.")

        if total_frames == 0:
            raise FaceSwapError("No frames extracted. Video may be corrupt.")

        # ── Source face ───────────────────────────────────────────────────
        upd("processing", 32, "Analysing source face…")
        source_face = await loop.run_in_executor(None, swapper.get_source_face, img_path)
        if source_face is None:
            raise FaceSwapError("No face detected in source image. Upload a clear frontal face photo.")

        # ── Similarity check ──────────────────────────────────────────────
        upd("processing", 34, "Checking face similarity…")
        # Sample the first extracted frame
        from pathlib import Path as _P
        sample_frames = sorted(_P(frames_dir).glob("frame_*.jpg"))
        sim_score = 0.0
        if sample_frames:
            mid_frame = str(sample_frames[min(5, len(sample_frames) - 1)])
            sim_score = await loop.run_in_executor(
                None, swapper.check_face_similarity, source_face, mid_frame
            )
        JOBS[job_id].update({"similarity_score": round(sim_score, 3)})

        # ── Face swap ─────────────────────────────────────────────────────
        processed_dir = str(FRAMES_DIR / f"{job_id}_out")
        stage_label   = "enhancing" if qmode != QualityMode.FAST else "processing"
        upd(stage_label, 36, f"Running face swap [{quality.upper()}] on {device}…")
        swapped, skipped = await loop.run_in_executor(
            None,
            swapper.process_video_optimized,
            source_face,
            frames_dir,
            processed_dir,
            qmode,
            face_index,
            None,          # max_frames (None = all)
            JOBS,
            job_id,
            36,            # progress_start
            78,            # progress_end
        )
        upd("rendering", 80, f"Swap complete — {swapped} faces swapped, {skipped} skipped.")

        if swapped == 0:
            raise FaceSwapError(
                "No faces found in any video frame. "
                "Ensure the target video contains a clearly visible face."
            )

        # ── Audio (skip for previews) ─────────────────────────────────────
        if is_preview:
            audio_path = None  # Don't attach audio to preview clips

        # ── Rebuild video ─────────────────────────────────────────────────
        kind_tag   = "preview" if is_preview else "output"
        out_file   = f"personaforge_{kind_tag}_{session_id[:8]}_{job_id[:8]}.mp4"
        out_path   = str(OUTPUTS_DIR / out_file)
        upd("rendering", 82, f"Encoding video (bitrate={bitrate})…")
        await loop.run_in_executor(
            None, rebuild_video, processed_dir, audio_path, out_path, fps,
            bitrate, mode == "cpu",   # cpu_mode flag for ultrafast preset
        )

        # ── Wrap up ───────────────────────────────────────────────────────
        size_mb   = get_file_size_mb(out_path)
        total_sec = time.perf_counter() - t_total
        JOBS[job_id].update({
            "status":       "done",
            "stage":        "done",
            "progress":     100,
            "message":      f"✓ Done in {total_sec:.1f}s! File: {size_mb:.1f} MB.",
            "output":       out_file,
            "file_size_mb": round(size_mb, 2),
        })
        logger.info("[%s] Pipeline done in %.1fs, output=%s", job_id[:8], total_sec, out_file)

    except FaceSwapError as e:
        JOBS[job_id].update({"status": "error", "stage": "error", "message": str(e)})
        logger.error("[%s] FaceSwapError: %s", job_id[:8], e)
    except Exception as e:
        JOBS[job_id].update({"status": "error", "stage": "error", "message": f"Unexpected error: {e}"})
        logger.exception("[%s] Unexpected error", job_id[:8])


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    try:
        check_models(auto_download=False)
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    print("\n" + "="*60)
    print("  PersonaForge AI — Starting…")
    print("  Open: http://127.0.0.1:8000")
    print("="*60 + "\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
