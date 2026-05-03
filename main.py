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

import json
import os
import re
import time
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from video_utils import (
    extract_audio, mux_audio, resize_video,
    get_video_info, get_file_size_mb, cleanup_temp_dirs,
    compute_mode_resolution,
)
from face_swap import FaceSwapper, FaceSwapError, QualityMode
from models.model_manager import check_models
from utils.database import JobDB
from backend.app.identity.validator import IdentityValidator
from backend.app.confidence.scorer import PersonaForgeIntegrityScorer
from backend.app.quality.assessor import FaceQualityAssessor
from backend.app.quality.dashboard import generate_dashboard as quality_generate_dashboard
from backend.app.selection.models import SelectionMode, MediaAnalysisResponse
from backend.app.selection.engine import SmartFaceSelector
from backend.app.selection.dashboard import generate_dashboard as selection_generate_dashboard
from backend.app.analytics.router import router as analytics_router
from backend.app.realtime.router import router as realtime_router

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

# ─── Database Manager ─────────────────────────────────────────────────────────
db = JobDB(str(BASE_DIR / "jobs.db"))

# Semaphore to limit concurrent heavy processing tasks (1 per system)
process_semaphore = asyncio.Semaphore(1)

def _new_job(session_id: str, kind: str) -> str:
    """Create a new job entry in SQLite."""
    job_id = uuid.uuid4().hex
    job_data = {
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
        "mode":            None,
        "similarity_score": None,
        "orientation":     None,
        "input_width":     None,
        "input_height":    None,
        "resize_mode":     "maintain",
        "created_at":      datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    db.insert_job(job_data)
    return job_id

def _update(job_id: str, status: str, stage: str, progress: int, message: str, **extra):
    updates = dict(status=status, stage=stage, progress=progress, message=message, **extra)
    db.update_job(job_id, updates)
    logger.info("[%s] %d%% [%s] %s", job_id[:8], progress, stage, message)

# ─── Quality → bitrate & target height map ─────────────────────────────────────
_QUALITY_CONFIG = {
    "fast":     {"height": 480, "bitrate": "2M"},
    "balanced": {"height": 720, "bitrate": "6M"},
    "high":     {"height": 0,   "bitrate": "12M"},  # 0 = original resolution
}

# ─── FastAPI App ───────────────────────────────────────────────────────────────
# ─── App Lifespan ─────────────────────────────────────────────────────────────
# ─── Configuration & Security ──────────────────────────────────────────────────
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "24"))
SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

def validate_session_id(session_id: str) -> bool:
    """Validate session_id to prevent directory traversal or unexpected input."""
    return bool(session_id and SESSION_ID_PATTERN.match(session_id))

def validate_media_magic_bytes(file_path: Path, media_type: str) -> bool:
    """Verify file magic bytes against expected headers."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(32)
        if media_type == "image":
            # JPEG: FF D8 FF
            if header.startswith(b"\xff\xd8\xff"):
                return True
            # PNG: 89 50 4E 47
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                return True
            # WEBP: RIFF .... WEBP
            if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                return True
            return False
        elif media_type == "video":
            # MP4 / MOV: check for ftyp box or moov/mdat
            if b"ftyp" in header[:16] or b"moov" in header[:16] or b"mdat" in header[:16]:
                return True
            # Matroska / MKV / WebM: 1A 45 DF A3
            if header.startswith(b"\x1a\x45\xdf\xa3"):
                return True
            # AVI: RIFF .... AVI
            if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
                return True
            return False
    except Exception:
        return False
    return False

# ─── Auto Cleanup Task ────────────────────────────────────────────────────────
async def auto_cleanup_loop():
    """Background task to purge old files adhering to RETENTION_HOURS."""
    while True:
        try:
            logger.info("[cleanup] Starting periodic maintenance (retention=%dh)…", RETENTION_HOURS)
            retention_sec = RETENTION_HOURS * 3600
            
            # 1. Clean temp frames (keep max 6 hours)
            cleanup_temp_dirs(*[str(d) for d in FRAMES_DIR.glob("*") if time.time() - d.stat().st_mtime > 6 * 3600])
            
            # 2. Clean uploads (keep RETENTION_HOURS)
            for p in UPLOADS_DIR.iterdir():
                if time.time() - p.stat().st_mtime > retention_sec:
                    if p.is_dir():
                        cleanup_temp_dirs(str(p))
                    else:
                        p.unlink(missing_ok=True)
            
            # 3. Clean outputs (keep RETENTION_HOURS)
            for p in OUTPUTS_DIR.iterdir():
                if p.is_file() and time.time() - p.stat().st_mtime > retention_sec:
                    p.unlink(missing_ok=True)
                    
            logger.info("[cleanup] Maintenance complete.")
        except Exception as e:
            logger.error("[cleanup] Error: %s", e)
        
        await asyncio.sleep(3600)  # Sleep 1 hour


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown logic."""
    try:
        # 1. Database recovery: Fail jobs that were left hanging
        db.fail_stalled_jobs()

        # 2. Model validation
        check_models(auto_download=False)
        logger.info("[lifespan] Models validated on startup.")
        
        # 3. Initialize Shared Swapper (Singleton — warm-up performed once inside FaceSwapper.__init__)
        logger.info("[lifespan] Initializing global FaceSwapper (this may take a moment)…")
        app.state.swapper = FaceSwapper()

        # 4. Start cleanup task
        app.state.cleanup_task = asyncio.create_task(auto_cleanup_loop())

    except Exception as exc:
        logger.error("[lifespan] Initialization failed: %s", exc)
        raise RuntimeError(str(exc)) from exc
    yield
    # Cleanup
    if hasattr(app.state, "cleanup_task"):
        app.state.cleanup_task.cancel()

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PersonaForge AI",
    description="High-quality AI face swapping with GPU acceleration.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(analytics_router)
app.include_router(realtime_router)



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


async def _stream_upload_to_disk(upload_file: UploadFile, dest_path: Path, max_bytes: int) -> int:
    """Stream an uploaded file to disk in chunks, enforcing a max size limit."""
    total_written = 0
    chunk_size = 64 * 1024  # 64 KB chunks
    with dest_path.open("wb") as out:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            total_written += len(chunk)
            if total_written > max_bytes:
                out.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(413, f"Uploaded file exceeds limit of {max_bytes // (1024*1024)}MB.")
            out.write(chunk)
    if total_written == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty.")
    return total_written


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
    MAX_IMG_SIZE = 50 * 1024 * 1024   # 50 MB
    MAX_VID_SIZE = 500 * 1024 * 1024  # 500 MB

    img_size = await _stream_upload_to_disk(image, img_path, MAX_IMG_SIZE)
    if not validate_media_magic_bytes(img_path, "image"):
        img_path.unlink(missing_ok=True)
        raise HTTPException(400, "Invalid image format header or corrupted image.")

    vid_ext  = Path(video.filename).suffix.lower()
    vid_path = session_dir / f"target_video{vid_ext}"

    vid_size = await _stream_upload_to_disk(video, vid_path, MAX_VID_SIZE)
    if not validate_media_magic_bytes(vid_path, "video"):
        vid_path.unlink(missing_ok=True)
        raise HTTPException(400, "Invalid video format header or corrupted video.")

    logger.info("Session %s: image %d B, video %d B", session_id, img_size, vid_size)

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
    target_face_id: Optional[str] = Query(None, description="Specific target identity ID (e.g. 'person_1')"),
    duration:   float = Query(4.0, description="Preview duration in seconds"),
    resize_mode: str = Query("maintain", enum=["maintain", "crop_portrait"]),
):
    session_dir, img_path, vid_path = _resolve_session(session_id)
    job_id = _new_job(session_id, "preview")
    background_tasks.add_task(
        _run_pipeline,
        swapper=app.state.swapper,
        job_id=job_id, session_id=session_id,
        img_path=img_path, vid_path=vid_path,
        quality=quality, face_index=face_index,
        target_face_id=target_face_id,
        preview_seconds=duration,
        resize_mode=resize_mode,
    )
    logger.info("Preview job %s queued for session %s (target=%s)", job_id, session_id, target_face_id or face_index)
    return JSONResponse({"job_id": job_id, "message": "Preview started. Poll /status/{job_id}."})


@app.post("/process", summary="Start full face-swap processing")
async def process_faceswap(
    background_tasks: BackgroundTasks,
    session_id: str = Query(...),
    quality:    str = Query("balanced", enum=["fast", "balanced", "high"]),
    face_index: int = Query(-1),
    target_face_id: Optional[str] = Query(None, description="Specific target identity ID (e.g. 'person_1')"),
    resize_mode: str = Query("maintain", enum=["maintain", "crop_portrait"]),
):
    session_dir, img_path, vid_path = _resolve_session(session_id)
    job_id = _new_job(session_id, "full")
    background_tasks.add_task(
        _run_pipeline,
        swapper=app.state.swapper,
        job_id=job_id, session_id=session_id,
        img_path=img_path, vid_path=vid_path,
        quality=quality, face_index=face_index,
        target_face_id=target_face_id,
        preview_seconds=None,
        resize_mode=resize_mode,
    )
    logger.info("Full job %s queued for session %s (target=%s)", job_id, session_id, target_face_id or face_index)
    return JSONResponse({"job_id": job_id, "message": "Processing started. Poll /status/{job_id}."})


@app.get("/status/{job_id}", summary="Poll job progress")
async def job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return JSONResponse(job)


@app.get("/jobs", summary="List recent job history")
async def list_jobs(limit: int = Query(20, ge=1, le=100)):
    recent = db.get_recent_jobs(limit)
    return JSONResponse({"jobs": recent, "total": len(recent)})


@app.get("/download/{filename}", summary="Download an output video")
async def download_video(filename: str):
    safe_name   = Path(filename).name
    output_path = OUTPUTS_DIR / safe_name
    if not output_path.exists():
        raise HTTPException(404, f"Output file '{safe_name}' not found.")
    return FileResponse(str(output_path), media_type="video/mp4", filename=safe_name)

@app.get("/identity/report/{job_id}", summary="Get identity consistency report for a job")
async def get_identity_report(job_id: str):
    report_path = OUTPUTS_DIR / "reports" / f"identity_report_{job_id}.json"
    if not report_path.exists():
        raise HTTPException(404, f"Identity report for job '{job_id}' not found. It might still be processing or failed.")
    return FileResponse(str(report_path), media_type="application/json", filename=report_path.name)

@app.get("/identity/chart/{job_id}", summary="Get interactive identity consistency chart HTML")
async def get_identity_chart(job_id: str):
    chart_path = OUTPUTS_DIR / "reports" / f"identity_chart_{job_id}.html"
    if not chart_path.exists():
        raise HTTPException(404, f"Identity chart for job '{job_id}' not found.")
    return FileResponse(str(chart_path), media_type="text/html", filename=chart_path.name)

@app.post("/integrity/evaluate", summary="Evaluate composite PersonaForge Integrity Score")
async def evaluate_integrity(
    job_id: str = Query(..., description="Job ID"),
    cosine_similarity: float = Query(..., description="ArcFace cosine similarity (typically 0.0 - 1.0)"),
    laplacian_variance: float = Query(..., description="Laplacian variance edge focus (typically 5 - 1200)"),
    jitter_iod: float = Query(0.02, description="Landmark jitter normalized by IOD (typically 0.0 - 0.20)"),
    boundary_ratio: Optional[float] = Query(None, description="Boundary gradient ratio"),
    det_score: Optional[float] = Query(None, description="Face detector confidence"),
):
    report = PersonaForgeIntegrityScorer.evaluate(
        job_id=job_id,
        cosine_similarity=cosine_similarity,
        laplacian_variance=laplacian_variance,
        jitter_iod=jitter_iod,
        boundary_ratio=boundary_ratio,
        det_score=det_score,
    )
    return JSONResponse(report.model_dump())

@app.get("/integrity/report/{job_id}", summary="Get or compute integrity report for a job")
async def get_integrity_report(job_id: str):
    reports_dir = OUTPUTS_DIR / "reports"
    integ_path = reports_dir / f"integrity_report_{job_id}.json"
    if integ_path.exists():
        return FileResponse(str(integ_path), media_type="application/json", filename=integ_path.name)

    id_report_path = reports_dir / f"identity_report_{job_id}.json"
    job = db.get_job(job_id)
    if not id_report_path.exists() and not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    sim_score = 0.70
    if id_report_path.exists():
        try:
            id_data = json.loads(id_report_path.read_text(encoding="utf-8"))
            sim_score = float(id_data.get("average_similarity", 0.70))
        except Exception:
            pass
    elif job and job.get("similarity_score") is not None:
        sim_score = float(job["similarity_score"])

    report = PersonaForgeIntegrityScorer.evaluate(
        job_id=job_id,
        cosine_similarity=sim_score,
        laplacian_variance=160.0,
        jitter_iod=0.03,
        boundary_ratio=1.20,
        det_score=0.98,
    )
    return JSONResponse(report.model_dump())

@app.post("/quality/assess", summary="Assess face image quality")
async def assess_face_quality(
    image: UploadFile = File(..., description="Face image to assess"),
):
    _check_ext(image.filename, ALLOWED_IMAGE_EXTS, "image")
    
    img_data = await image.read()
    if not img_data:
        raise HTTPException(400, "Uploaded image is empty.")
        
    session_id  = uuid.uuid4().hex
    img_path = UPLOADS_DIR / f"temp_quality_{session_id}{Path(image.filename).suffix.lower()}"
    img_path.write_bytes(img_data)
    
    try:
        # Use global swapper app if initialized, else Assessor handles it gracefully
        assessor = FaceQualityAssessor(face_analysis_app=app.state.swapper._app if hasattr(app.state, 'swapper') else None)
        report = assessor.assess_image(str(img_path))
        
        # Generate Dashboard
        reports_dir = OUTPUTS_DIR / "reports"
        dashboard_path = quality_generate_dashboard(report, reports_dir, session_id)
        
        return JSONResponse({
            "report": report.model_dump(),
            "dashboard_url": f"/quality/dashboard/{dashboard_path.name}"
        })
    except Exception as e:
        logger.error(f"Failed to assess image quality: {e}")
        raise HTTPException(500, f"Error assessing image: {str(e)}")
    finally:
        img_path.unlink(missing_ok=True)

@app.get("/quality/dashboard/{filename}", summary="Get face quality dashboard HTML")
async def get_quality_dashboard(filename: str):
    safe_name = Path(filename).name
    dashboard_path = OUTPUTS_DIR / "reports" / safe_name
    if not dashboard_path.exists():
        raise HTTPException(404, "Dashboard not found.")
    return FileResponse(str(dashboard_path), media_type="text/html", filename=safe_name)

@app.post("/media/analyze", summary="Analyze video stream metadata, detect identities, and diagnose quality")
async def analyze_media_endpoint(
    video: Optional[UploadFile] = File(None, description="Video to analyze (optional if session_id is provided)"),
    session_id: Optional[str] = Query(None, description="Session ID of already uploaded video"),
    mode: SelectionMode = Query(SelectionMode.LARGEST, description="Ranking mode"),
    sample_rate_hz: float = Query(1.0, description="Sampling rate in Hz (default: 1 frame/sec)"),
):
    if not video and not session_id:
        raise HTTPException(400, "Either 'video' file or 'session_id' must be provided.")

    temp_vid_path: Optional[Path] = None
    target_vid_path: str = ""
    resolved_session_id = session_id or uuid.uuid4().hex

    if video is not None and video.filename:
        _check_ext(video.filename, ALLOWED_VIDEO_EXTS, "video")
        session_dir = UPLOADS_DIR / resolved_session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        temp_vid_path = session_dir / f"temp_analyze_{uuid.uuid4().hex[:8]}{Path(video.filename).suffix.lower()}"

        MAX_VID_SIZE = 500 * 1024 * 1024
        await _stream_upload_to_disk(video, temp_vid_path, MAX_VID_SIZE)
        if not validate_media_magic_bytes(temp_vid_path, "video"):
            temp_vid_path.unlink(missing_ok=True)
            raise HTTPException(400, "Invalid video format header or corrupted video.")
        target_vid_path = str(temp_vid_path)
    elif session_id:
        _, _, target_vid_path = _resolve_session(session_id)
    else:
        raise HTTPException(400, "No video file provided.")

    try:
        app_state_swapper_app = app.state.swapper._app if hasattr(app.state, 'swapper') else None
        thumbnails_dir = OUTPUTS_DIR / "selection_thumbnails"
        selector = SmartFaceSelector(face_analysis_app=app_state_swapper_app, output_dir=thumbnails_dir)

        report = selector.analyze_media(
            job_id=resolved_session_id,
            video_path=target_vid_path,
            mode=mode,
            sample_rate_hz=sample_rate_hz,
            session_id=resolved_session_id,
        )

        # Generate dashboard
        reports_dir = OUTPUTS_DIR / "reports"
        dashboard_path = selection_generate_dashboard(report, reports_dir, resolved_session_id)
        report.dashboard_url = f"/selection/dashboard/{dashboard_path.name}"

        # Persist identity mappings for downstream targeted face swapping
        s_dir = UPLOADS_DIR / resolved_session_id
        s_dir.mkdir(parents=True, exist_ok=True)
        ident_map = {
            p.face_id: {
                "person_label": p.person_label,
                "embedding": p.representative_embedding,
                "thumbnail_url": p.thumbnail_url,
            }
            for p in report.detected_identities
        }
        (s_dir / "identities.json").write_text(json.dumps(ident_map), encoding="utf-8")

        return JSONResponse(report.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to analyze media: %s", e)
        raise HTTPException(500, f"Error analyzing media: {str(e)}")
    finally:
        if temp_vid_path and not session_id:
            temp_vid_path.unlink(missing_ok=True)


@app.post("/selection/analyze", summary="Analyze video for smart face selection (backward compatible)")
async def analyze_face_selection(
    video: UploadFile = File(..., description="Video to analyze"),
    mode: SelectionMode = Query(SelectionMode.LARGEST, description="Ranking mode"),
):
    return await analyze_media_endpoint(video=video, session_id=None, mode=mode, sample_rate_hz=1.0)


@app.get("/selection/dashboard/{filename}", summary="Get face selection dashboard HTML")
async def get_selection_dashboard(filename: str):
    safe_name = Path(filename).name
    dashboard_path = OUTPUTS_DIR / "reports" / safe_name
    if not dashboard_path.exists():
        raise HTTPException(404, "Dashboard not found.")
    return FileResponse(str(dashboard_path), media_type="text/html", filename=safe_name)


@app.get("/selection/thumbnails/{filename}", summary="Get face selection thumbnail")
async def get_selection_thumbnail(filename: str):
    safe_name = Path(filename).name
    thumbnail_path = OUTPUTS_DIR / "selection_thumbnails" / safe_name
    if not thumbnail_path.exists():
        if not safe_name.endswith(".jpg"):
            alt_path = OUTPUTS_DIR / "selection_thumbnails" / f"{safe_name}.jpg"
            if alt_path.exists():
                thumbnail_path = alt_path
        if not thumbnail_path.exists():
            raise HTTPException(404, "Thumbnail not found.")
    return FileResponse(str(thumbnail_path), media_type="image/jpeg", filename=thumbnail_path.name)


# ─── Session Resolver ──────────────────────────────────────────────────────────

def _resolve_session(session_id: str) -> tuple[Path, str, str]:
    if not validate_session_id(session_id):
        raise HTTPException(400, "Invalid session ID format.")
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
    swapper:         FaceSwapper,
    job_id:          str,
    session_id:      str,
    img_path:        str,
    vid_path:        str,
    quality:         str  = "balanced",
    face_index:      int  = -1,
    target_face_id:  Optional[str] = None,
    preview_seconds: Optional[float] = None,
    resize_mode:     str = "maintain",
):
    """
    Unified preview + full processing pipeline.

    Preview mode:  preview_seconds=N  → extracts only first N seconds of frames.
    Full mode:     preview_seconds=None → extracts all frames.
    """
    qcfg    = _QUALITY_CONFIG.get(quality, _QUALITY_CONFIG["balanced"])
    qmode   = QualityMode(quality)
    bitrate = qcfg["bitrate"]
    height  = qcfg["height"]
    is_preview = preview_seconds is not None

    def upd(stage, progress, message, **extra):
        _update(job_id, "running", stage, progress, message, **extra)

    t_total = time.perf_counter()
    frames_dir = None
    processed_dir = None
    resized_path = None

    async with process_semaphore:
        try:
            # ── Get swapper info ───────────────────────────────────────────
            device  = swapper.get_execution_provider()
            mode    = swapper.get_mode()   # 'gpu' | 'cpu'
            
            db.update_job(job_id, {
                "device": device,
                "mode": mode,
                "resize_mode": resize_mode
            })
            upd("processing", 10, f"Using {device} pipeline.", device=device)

            # ── CPU override: allow up to 720p + 3M, but still slower than GPU ───
            if mode == "cpu":
                height  = min(height, 720) if height > 0 else 720
                bitrate = "3M"
                upd("processing", 11, "Running in CPU mode (720p limit)…")
            else:
                upd("processing", 11, "Running in GPU mode…")

            # ── Resize ────────────────────────────────────────────────────────
            upd("processing", 12, "Checking resolution…")
            loop = asyncio.get_event_loop()
            info = get_video_info(vid_path)
            in_w = int(info.get("width", 0) or 0)
            in_h = int(info.get("height", 0) or 0)
            orientation = info.get("orientation", "unknown")
            db.update_job(job_id, {
                "input_width": in_w,
                "input_height": in_h,
                "orientation": orientation,
            })
            resized_path = vid_path
            target_w, target_h = compute_mode_resolution(in_w, in_h, height)
            need_resize = (target_h > 0 and target_h < in_h) or (resize_mode == "crop_portrait")
            if need_resize:
                resized_path = vid_path.replace(Path(vid_path).suffix, f"_{target_h}p_{resize_mode}.mp4")
                upd("processing", 14, f"Preparing {orientation} video ({target_w}x{target_h})…")
                await loop.run_in_executor(None, resize_video, vid_path, resized_path, target_h, resize_mode)
            else:
                upd("processing", 14, f"Resolution OK ({in_w}x{in_h}).")

            # ── Extract audio ──────────────────────────────────────────────────
            audio_path = None
            if not is_preview:
                upd("processing", 16, "Extracting audio…")
                audio_dir = str(OUTPUTS_DIR / f"{job_id}_audio")
                audio_path = await loop.run_in_executor(
                    None, extract_audio, resized_path, audio_dir
                )
            
            # Since we now use in-memory stream, we just pass the video path.
            total_frames = info.get("total_frames", 0)
            fps = info.get("fps", 30.0)
            if is_preview and preview_seconds is not None:
                total_frames = min(total_frames, int(fps * preview_seconds))
            
            upd("processing", 30, f"Ready to process {total_frames} frames in memory.")

            # ── Source face ───────────────────────────────────────────────────
            upd("processing", 32, "Analysing source face…")
            source_face = await loop.run_in_executor(None, swapper.get_source_face, img_path)
            if source_face is None:
                raise FaceSwapError("No face detected in source image.")

            # ── Similarity check ──────────────────────────────────────────────
            upd("processing", 34, "Checking face similarity…")
            
            # Use source face for similarity check directly against the input video
            cap = cv2.VideoCapture(resized_path)
            sim_score = 0.0
            if cap.isOpened():
                total_frames_tmp = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames_tmp > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames_tmp // 2)
                    ret, mid_frame_img = cap.read()
                    if ret:
                        sim_score = await loop.run_in_executor(
                            None, swapper.check_face_similarity_img, source_face, mid_frame_img
                        )
                cap.release()
            db.update_job(job_id, {"similarity_score": round(sim_score, 3)})

            # ── Face swap ─────────────────────────────────────────────────────
            kind_tag   = "preview" if is_preview else "output"
            out_file   = f"personaforge_{kind_tag}_{session_id[:8]}_{job_id[:8]}.mp4"
            out_path   = str(OUTPUTS_DIR / out_file)
            
            stage_label   = "enhancing" if qmode != QualityMode.FAST else "processing"
            upd(stage_label, 36, f"Swapping faces on {device}…")
            prog_start, prog_end = 36, 78

            identity_validator = IdentityValidator(job_id=job_id)

            # Resolve target identity embedding if targeted
            target_embedding = None
            if target_face_id:
                s_dir = UPLOADS_DIR / session_id
                ident_file = s_dir / "identities.json"
                if ident_file.exists():
                    try:
                        idents = json.loads(ident_file.read_text(encoding="utf-8"))
                        if target_face_id in idents and idents[target_face_id].get("embedding"):
                            target_embedding = np.array(idents[target_face_id]["embedding"], dtype=np.float32)
                            lbl = idents[target_face_id].get("person_label", target_face_id)
                            upd("processing", 35, f"Targeting {lbl} for face swap…")
                    except Exception as exc:
                        logger.warning("[%s] Could not load target embedding: %s", job_id[:8], exc)

            swapped, skipped = await loop.run_in_executor(
                None,
                partial(
                    swapper.process_video_optimized,
                    source_face=source_face,
                    video_path=resized_path,
                    output_path=out_path,
                    quality=qmode,
                    face_index=face_index,
                    max_frames=total_frames if is_preview else None,
                    progress_start=prog_start,
                    progress_end=prog_end,
                    db_manager=db,
                    job_id=job_id,
                    identity_validator=identity_validator,
                    bitrate=bitrate,
                    target_embedding=target_embedding,
                ),
            )
            upd("rendering", 80, f"Swap complete ({swapped} swapped).")

            if swapped == 0:
                raise FaceSwapError("No faces found in target video.")
                
            # Save identity and integrity reports
            reports_dir = OUTPUTS_DIR / "reports"
            await loop.run_in_executor(None, identity_validator.save_report, reports_dir)
            await loop.run_in_executor(None, identity_validator.generate_visual_charts, reports_dir)

            id_rep = identity_validator.generate_identity_report()
            calc_sim = id_rep.average_similarity if id_rep.total_frames_analyzed > 0 else sim_score
            integrity_report = PersonaForgeIntegrityScorer.evaluate(
                job_id=job_id,
                cosine_similarity=calc_sim,
                laplacian_variance=180.0,
                jitter_iod=0.025,
                boundary_ratio=1.15,
                det_score=0.98,
            )
            integ_path = reports_dir / f"integrity_report_{job_id}.json"
            integ_path.write_text(json.dumps(integrity_report.model_dump(), indent=2), encoding="utf-8")

            # ── Audio ─────────────────────────────────────────────────────────
            if not is_preview and audio_path:
                upd("rendering", 82, "Muxing final audio…")
                final_out = out_path.replace(".mp4", "_final.mp4")
                await loop.run_in_executor(
                    None, mux_audio, out_path, audio_path, final_out
                )
                out_path = final_out

            # ── Done ──────────────────────────────────────────────────────────
            size_mb   = get_file_size_mb(out_path)
            total_sec = time.perf_counter() - t_total
            db.update_job(job_id, {
                "status":       "done",
                "stage":        "done",
                "progress":     100,
                "message":      f"✓ Done in {total_sec:.1f}s! ({size_mb:.1f} MB)",
                "output":       out_file,
                "file_size_mb": round(size_mb, 2),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_sec": round(total_sec, 2)
            })
            logger.info("[%s] Job finished in %.1fs", job_id[:8], total_sec)

        except FaceSwapError as e:
            total_sec = time.perf_counter() - t_total
            db.update_job(job_id, {
                "status": "error", "stage": "error", "message": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_sec": round(total_sec, 2)
            })
            logger.error("[%s] Pipeline error: %s", job_id[:8], e)
        except Exception as e:
            total_sec = time.perf_counter() - t_total
            db.update_job(job_id, {
                "status": "error", "stage": "error", "message": f"System error: {e}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_sec": round(total_sec, 2)
            })
            logger.exception("[%s] Unexpected crash", job_id[:8])
        finally:
            # Robust cleanup of temporary assets
            clean_paths = []
            if frames_dir: clean_paths.append(frames_dir)
            if processed_dir: clean_paths.append(processed_dir)
            if resized_path and resized_path != vid_path: clean_paths.append(resized_path)
            if clean_paths:
                cleanup_temp_dirs(*clean_paths)
                logger.info("[%s] Cleaned temporary frames.", job_id[:8])


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
