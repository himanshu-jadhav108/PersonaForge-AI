"""
face_swap.py — PersonaForge AI · Face Detection & Swap Engine

Features:
  - QualityMode: FAST / BALANCED / HIGH with per-mode enhancement
  - Multi-face: swap all or a specific face by index
  - Smart blending: seamlessClone for realistic boundary blending
  - Face similarity check: warns if source/target faces are very different
  - Preview mode: process only the first N frames
  - KCF face tracking between detections to cut GPU load
  - Threaded JPEG writes for async disk I/O
  - Dual pipeline: GPU path unchanged; CPU path uses pipeline_cpu (optimised)
"""

import logging
import time
from pathlib import Path
import cv2
import numpy as np
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from models.model_manager import get_model_path, MODEL_CONFIG
from utils.tracker_factory import make_tracker
from backend.app.models.factory import ModelFactory
from video_utils import get_video_info, get_ffmpeg_writer

logger = logging.getLogger("personaforge.face_swap")

# Avoid printing the same CUDA fallback notice on every new job.
_cuda_fallback_notified = False

# ── Quality Mode ──────────────────────────────────────────────────────────────

class QualityMode(str, Enum):
    FAST     = "fast"
    BALANCED = "balanced"
    HIGH     = "high"

# Per-mode JPEG quality for intermediate frames
_JPEG_QUALITY = {
    QualityMode.FAST:     88,
    QualityMode.BALANCED: 95,
    QualityMode.HIGH:     99,
}

# ── Tuning Constants ──────────────────────────────────────────────────────────
DETECT_EVERY_N_FRAMES = 5      # Full detection every N frames; track in between
FACE_CROP_PADDING     = 0.35   # Padding around detected face bbox
WRITE_WORKERS         = 4      # Threads for concurrent JPEG writes
BATCH_FLUSH_SIZE      = 32     # Flush write queue after this many frames

# ── Custom Exception ──────────────────────────────────────────────────────────

class FaceSwapError(Exception):
    """User-facing face swap failures."""
    pass

def _find_model(name: str) -> str:
    config_name = name.removesuffix(".onnx") if name.endswith(".onnx") else name
    if config_name not in MODEL_CONFIG:
        config_name = "inswapper"
    path = get_model_path(config_name)
    if path.exists():
        return str(path)
    model_url = MODEL_CONFIG[config_name]["url"]
    raise FileNotFoundError(
        f"Model '{path.name}' not found.\n"
        f"Please download it from: {model_url}\n"
        "and place it inside the models/ directory."
    )

# ── Tracker Factory ───────────────────────────────────────────────────────────

def _make_tracker():
    return make_tracker("GPU")

# ────────────────────────────────────────────────────────────────────────────────
# FaceSwapper
# ────────────────────────────────────────────────────────────────────────────────

class FaceSwapper:
    """
    PersonaForge AI face-swap engine.

    Key features:
      • GPU inference via CUDAExecutionProvider
      • KCF tracking to skip face detection on most frames
      • Crop-and-paste: swap only the face ROI (faster GPU utilisation)
      • Seamless clone blending for realistic boundary merging
      • Quality modes controlling detection resolution and enhancement
      • Multi-face or single-face targeting
      • Preview mode (process first N frames only)
    """

    def __init__(self, model_name: str = "inswapper_128.onnx", use_gpu: bool = True):
        self._app       = None
        self._swap_adapter = None
        self._providers = ["CPUExecutionProvider"]
        self._mode      = "cpu"   # set properly in _load()
        self._load(model_name, use_gpu)

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load(self, model_name: str, use_gpu: bool) -> None:
        global _cuda_fallback_notified
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError("insightface not installed. Run: pip install insightface onnxruntime-gpu")

        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            logger.info("ONNX providers available: %s", available)
            if use_gpu and "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("GPU (CUDA) selected.")
            else:
                if use_gpu:
                    if not _cuda_fallback_notified:
                        logger.info("CUDAExecutionProvider not found; falling back to CPU.")
                        _cuda_fallback_notified = True
                providers = ["CPUExecutionProvider"]
        except ImportError:
            providers = ["CPUExecutionProvider"]

        # ── Determine hardware mode ────────────────────────────────────────
        from config import config_cpu as _ccpu
        from config import config_gpu as _cgpu
        _is_gpu = "CUDAExecutionProvider" in providers
        self._mode = "gpu" if _is_gpu else "cpu"
        _cfg = _cgpu if _is_gpu else _ccpu
        _det_size = _cfg.DET_SIZE
        _banner   = _cfg.STARTUP_MSG
        logger.info("%s", _banner)
        print(f"\n  ✦ PersonaForge AI — {_banner}\n")

        logger.info("Loading InsightFace buffalo_l …")
        t0 = time.perf_counter()
        try:
            self._app = FaceAnalysis(name="buffalo_l", providers=providers)
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            self._app.prepare(ctx_id=ctx_id, det_size=_det_size)
            self._providers = providers
        except Exception as e:
            if "load_library" in str(e).lower() or "dll" in str(e).lower():
                logger.error("CUDA DLL missing — falling back to CPU.")
                providers = ["CPUExecutionProvider"]
                self._mode = "cpu"
                self._app = FaceAnalysis(name="buffalo_l", providers=providers)
                self._app.prepare(ctx_id=-1, det_size=_ccpu.DET_SIZE)
                self._providers = providers
                print(f"\n  ✦ PersonaForge AI — {_ccpu.STARTUP_MSG}\n")
            else:
                raise
        logger.info("InsightFace loaded in %.2fs", time.perf_counter() - t0)

        model_path = _find_model(model_name)
        logger.info("Loading swap model from '%s' …", model_path)
        t0 = time.perf_counter()
        try:
            self._swap_adapter = ModelFactory.get_model(model_name)
            self._swap_adapter.load_model(model_path, self._providers)
            logger.info("Swap model adapter loaded in %.2fs (%s)", time.perf_counter() - t0, self._providers[0])
        except Exception as e:
            logger.error("Failed to load swap model adapter: %s", e)
            raise

        # ─── Warm up ──────────────────────────────────────────────────────────
        self._warm_up()

    def _warm_up(self) -> None:
        """Perform a dummy inference to warm up the ONNX/CUDA engine."""
        logger.info("Warming up AI engine…")
        try:
            # Create a small dummy face and frame
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            # Detect (even if it finds nothing, it warms up the detector)
            self._app.get(dummy_frame)
            logger.info("AI engine warmed up successfully.")
        except Exception as e:
            logger.warning("Warm-up failed (non-critical): %s", e)

    def get_execution_provider(self) -> str:
        return "GPU" if "CUDAExecutionProvider" in self._providers else "CPU"

    def get_mode(self) -> str:
        """Return 'gpu' or 'cpu' based on the detected execution provider."""
        return self._mode

    # ── Optimised Dispatcher ───────────────────────────────────────────────────

    def process_video_optimized(
        self,
        source_face,
        video_path:  str,
        output_path:  str,
        quality:     "QualityMode" = None,
        face_index:  int          = -1,
        max_frames:  Optional[int] = None,
        progress_start: int       = 36,
        progress_end:   int       = 78,
        db_manager                = None,
        job_id:      str          = None,
        identity_validator        = None,
    ) -> tuple[int, int]:
        """
        Route to GPU or CPU pipeline based on detected hardware.

        GPU  → delegates to process_video() (original, unchanged).
        CPU  → delegates to pipeline_cpu.process_video_cpu() (optimised).
        """
        if self._mode == "gpu":
            from pipelines.pipeline_gpu import process_video_gpu
            return process_video_gpu(
                swapper        = self,
                source_face    = source_face,
                video_path     = video_path,
                output_path     = output_path,
                quality        = quality if quality is not None else QualityMode.BALANCED,
                face_index     = face_index,
                max_frames     = max_frames,
                progress_start = progress_start,
                progress_end   = progress_end,
                db_manager     = db_manager,
                job_id         = job_id,
                identity_validator = identity_validator,
            )
        else:
            from pipelines.pipeline_cpu import process_video_cpu
            return process_video_cpu(
                swapper_app    = self._app,
                swap_adapter   = self._swap_adapter,
                source_face    = source_face,
                video_path     = video_path,
                output_path     = output_path,
                face_index     = face_index,
                max_frames     = max_frames,
                progress_start = progress_start,
                progress_end   = progress_end,
                db_manager     = db_manager,
                job_id         = job_id,
                identity_validator = identity_validator,
            )

    # ── Source Face ────────────────────────────────────────────────────────────

    def get_source_face(self, image_path: str):
        """Detect and return the largest face from the source image."""
        img = cv2.imread(image_path)
        if img is None:
            raise FaceSwapError(f"Cannot read source image: '{image_path}'")
        faces = self._app.get(img)
        if not faces:
            return None
        best = max(faces, key=lambda f: _bbox_area(f.bbox))
        logger.info("Source face area: %d px²", int(_bbox_area(best.bbox)))
        return best

    # ── Face Similarity Check ─────────────────────────────────────────────────

    def check_face_similarity(self, source_face, frame_path: str) -> float:
        """
        Compute cosine similarity between source face embedding and the
        largest face in a sample frame. Returns 0.0 if no face found.
        Score < 0.2 suggests a significant mismatch (warn user).
        """
        frame = cv2.imread(frame_path)
        if frame is None:
            return 0.0
        faces = self._app.get(frame)
        if not faces:
            return 0.0
        target = max(faces, key=lambda f: _bbox_area(f.bbox))
        try:
            src_emb = np.array(source_face.embedding, dtype=np.float32)
            tgt_emb = np.array(target.embedding,      dtype=np.float32)
            score = float(np.dot(src_emb, tgt_emb) /
                          (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-9))
            logger.info("Face similarity score: %.3f", score)
            return score
        except Exception as e:
            logger.debug("Similarity check failed: %s", e)
            return 0.0

    def check_face_similarity_img(self, source_face, frame: np.ndarray) -> float:
        """
        Compute cosine similarity directly from a BGR numpy array.
        """
        if frame is None:
            return 0.0
        faces = self._app.get(frame)
        if not faces:
            return 0.0
        target = max(faces, key=lambda f: _bbox_area(f.bbox))
        try:
            src_emb = np.array(source_face.embedding, dtype=np.float32)
            tgt_emb = np.array(target.embedding,      dtype=np.float32)
            score = float(np.dot(src_emb, tgt_emb) /
                          (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-9))
            logger.info("Face similarity score: %.3f", score)
            return score
        except Exception as e:
            logger.debug("Similarity check failed: %s", e)
            return 0.0

    # ── Video Processing Pipeline ──────────────────────────────────────────────

    def process_video(
        self,
        source_face,
        video_path:  str,
        output_path: str,
        quality:     QualityMode = QualityMode.BALANCED,
        face_index:  int         = -1,   # -1 = all faces
        max_frames:  Optional[int] = None,  # None = all; N = preview mode
        progress_start: int      = 36,
        progress_end:   int      = 78,
        db_manager                = None,
        job_id:      str         = None,
        identity_validator       = None,
    ) -> tuple[int, int]:
        """
        Core processing loop.
        Now uses in-memory video streaming.
        """
        info = get_video_info(video_path)
        total = info.get("total_frames", 0)
        fps = info.get("fps", 30.0)
        orig_w = info.get("width", 0)
        orig_h = info.get("height", 0)

        if total == 0:
            raise RuntimeError(f"Could not read video info for '{video_path}'")
            
        if max_frames is not None:
            total = min(total, max_frames)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video '{video_path}'")
            
        writer = get_ffmpeg_writer(
            output_path=output_path,
            fps=fps,
            width=orig_w,
            height=orig_h,
            bitrate="6M",
            cpu_mode=False,
        )

        swapped  = skipped = 0
        tracker  = None
        tracked_bbox = None

        t_start = time.perf_counter()

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            h, w     = frame.shape[:2]
            face_found = False

            # ── Face Detection / Tracking ──────────────────────────────────
            run_detection = (i % DETECT_EVERY_N_FRAMES == 0) or tracker is None

            if not run_detection and tracker is not None:
                ok, bbox = tracker.update(frame)
                if ok:
                    tracked_bbox = tuple(int(v) for v in bbox)
                    face_found   = True
                else:
                    tracker       = None
                    run_detection = True

            if run_detection:
                all_faces = self._app.get(frame)
                if all_faces:
                    # Sort by area descending
                    all_faces.sort(key=lambda f: _bbox_area(f.bbox), reverse=True)
                    best  = all_faces[0]
                    x1, y1, x2, y2 = [int(v) for v in best.bbox[:4]]
                    bw, bh = x2 - x1, y2 - y1
                    tracked_bbox = (x1, y1, bw, bh)
                    face_found   = True
                    tracker      = _make_tracker()
                    if tracker is not None:
                        tracker.init(frame, tracked_bbox)
                else:
                    tracker      = None
                    tracked_bbox = None

            # ── Crop-based Swap ────────────────────────────────────────────
            if face_found and tracked_bbox and source_face is not None:
                x, y, bw, bh = tracked_bbox
                pad_x = int(bw * FACE_CROP_PADDING)
                pad_y = int(bh * FACE_CROP_PADDING)
                x1c = max(0, x - pad_x)
                y1c = max(0, y - pad_y)
                x2c = min(w, x + bw + pad_x)
                y2c = min(h, y + bh + pad_y)

                crop = frame[y1c:y2c, x1c:x2c]

                crop_faces = self._app.get(crop)
                if crop_faces:
                    # Filter by face_index (-1 = all)
                    crop_faces.sort(key=lambda f: _bbox_area(f.bbox), reverse=True)
                    targets = (
                        crop_faces if face_index == -1
                        else ([crop_faces[face_index]] if face_index < len(crop_faces) else crop_faces)
                    )

                    result_crop = crop.copy()
                    did_swap    = False
                    for tf in targets:
                        try:
                            result_crop = self._swap_adapter.swap_face(result_crop, tf, source_face)
                            did_swap    = True
                        except Exception as e:
                            logger.debug("Swap on crop failed: %s", e)

                    if did_swap:
                        # Quality-driven enhancement on the crop only
                        result_crop = _enhance_crop(result_crop, quality)

                        # Build mask for seamless clone blending
                        result = _blend_crop(frame, result_crop, x1c, y1c, x2c, y2c)
                        swapped += 1
                        
                        if identity_validator:
                            swapped_faces = self._app.get(result_crop)
                            if swapped_faces:
                                swapped_faces.sort(key=lambda f: _bbox_area(f.bbox), reverse=True)
                                swapped_face = swapped_faces[0]
                                timestamp = float(i)
                                identity_validator.add_record(i, timestamp, source_face.embedding, swapped_face.embedding)
                    else:
                        result = frame
                        skipped += 1
                else:
                    result  = frame
                    skipped += 1
            else:
                result  = frame
                skipped += 1

            # ── Direct Pipe Write ──────────────────────────────────────────
            writer.stdin.write(result.tobytes())

            # ── Progress ───────────────────────────────────────────────────
            if db_manager is not None and job_id is not None:
                span = progress_end - progress_start
                pct  = progress_start + int((i + 1) / total * span)
                db_manager.update_job(job_id, {
                    "progress": pct,
                    "message": f"Frame {i+1}/{total} — swapped={swapped}, skipped={skipped}"
                })

        # Clean up
        cap.release()
        writer.stdin.close()
        writer.wait()

        elapsed = time.perf_counter() - t_start
        fps_out = total / elapsed if elapsed > 0 else 0
        logger.info(
            "Processing done in %.1fs → %.1f fps | swapped=%d skipped=%d",
            elapsed, fps_out, swapped, skipped,
        )
        return swapped, skipped


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float((x2 - x1) * (y2 - y1)))


def _enhance_crop(img: np.ndarray, quality: QualityMode) -> np.ndarray:
    """Apply quality-appropriate enhancement on the face crop."""
    if quality == QualityMode.FAST:
        return img  # No enhancement
    elif quality == QualityMode.BALANCED:
        # Mild unsharp mask
        blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
        return cv2.addWeighted(img, 1.3, blurred, -0.3, 0)
    else:  # HIGH
        # Advanced bilateral filter to smooth + stronger unsharp mask for clarity
        smooth = cv2.bilateralFilter(img, d=7, sigmaColor=65, sigmaSpace=65)
        blurred = cv2.GaussianBlur(smooth, (0, 0), 2)
        return cv2.addWeighted(smooth, 1.6, blurred, -0.6, 0)


def _blend_crop(
    frame: np.ndarray,
    result_crop: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    """
    Blend the swapped crop back into the frame using seamlessClone
    for a smooth, realistic boundary. Falls back to direct paste on error.
    """
    result = frame.copy()
    crop_h = y2 - y1
    crop_w = x2 - x1

    if result_crop.shape[0] != crop_h or result_crop.shape[1] != crop_w:
        # Shape mismatch — resize to fit
        result_crop = cv2.resize(result_crop, (crop_w, crop_h))

    try:
        # Elliptical mask for seamless clone
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cy, cx = crop_h // 2, crop_w // 2
        axes   = (max(1, crop_w // 2 - 4), max(1, crop_h // 2 - 4))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

        center = (x1 + cx, y1 + cy)
        result = cv2.seamlessClone(result_crop, result, mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        logger.debug("seamlessClone failed (%s) — using direct paste.", e)
        result[y1:y2, x1:x2] = result_crop

    return result


def _write_frame(path: str, frame: np.ndarray, quality: int = 95) -> None:
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
