"""
pipeline_cpu.py — PersonaForge AI · CPU-Optimized Processing Pipeline

This module contains the CPU-mode face-swap loop. It is called exclusively
when CUDAExecutionProvider is NOT available. The GPU pipeline in face_swap.py
is completely unmodified — this is an additive module only.

CPU optimisations applied here:
  • Pre-resize every frame to TARGET_HEIGHT (480p) before detection + inference
  • Frame skipping: process 1 in PROCESS_EVERY_N_FRAMES frames, reuse last result
  • Face detection every DETECT_EVERY processed frames; KCF tracking between
  • No enhancement pass (ENHANCEMENT_ENABLED = False)
  • Direct-paste blending (no seamlessClone — saves ~40 ms/frame on CPU)
  • Frame reuse: skip inference if tracked face centre barely moved
  • Throttled progress updates to database
"""

import logging
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from config import config_cpu as cfg
from utils.tracker_factory import make_tracker
from backend.app.tracking.factory import get_tracker
from pipelines.blending.factory import get_blender
from video_utils import get_video_info, get_ffmpeg_writer

logger = logging.getLogger("personaforge.pipeline_cpu")


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float((x2 - x1) * (y2 - y1)))


def _centre(bbox_xywh: tuple) -> tuple[float, float]:
    x, y, w, h = bbox_xywh
    return (x + w / 2.0, y + h / 2.0)


def _make_tracker(tracker_type: str = "kcf"):
    return get_tracker(tracker_type)


def _write_frame_pipe(writer, frame: np.ndarray) -> None:
    writer.stdin.write(frame.tobytes())


def _direct_paste(
    frame: np.ndarray,
    crop: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    """Paste the swapped crop back using AlphaBlend (CPU mode)."""
    return get_blender("alpha").blend(frame, crop, x1, y1, x2, y2)


# ─── Public API ────────────────────────────────────────────────────────────────

def process_video_cpu(
    swapper_app,            # insightface FaceAnalysis instance
    swap_adapter,           # InSwapperAdapter (or duck-typed equivalent)
    source_face,            # insightface Face object for source
    video_path:     str,
    output_path:    str,
    quality                 = None,     # QualityMode enum or string
    face_index:     int     = -1,
    max_frames:     Optional[int] = None,
    progress_start: int     = 36,
    progress_end:   int     = 78,
    db_manager              = None,
    job_id:         Optional[str] = None,
    identity_validator      = None,
    bitrate:        Optional[str] = None,
) -> tuple[int, int]:
    """
    CPU face-swap processing pipeline.
    Streams frames directly to ffmpeg writer via subprocess pipe.
    Returns (swapped_frames_count, skipped_frames_count).
    """
    FACE_CROP_PADDING = 0.50

    info = get_video_info(video_path)
    total  = info.get("total_frames", 0)
    fps    = info.get("fps", 30.0)
    orig_w = info.get("width", 0)
    orig_h = info.get("height", 0)

    if total == 0:
        raise RuntimeError(f"Could not read video info for '{video_path}'")

    if max_frames is not None:
        total = min(total, max_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video '{video_path}'")
        
    chosen_bitrate = bitrate or cfg.BITRATE or "3M"
    # Open FFMPEG pipe writer
    writer = get_ffmpeg_writer(
        output_path=output_path,
        fps=fps,
        width=orig_w,
        height=orig_h,
        bitrate=chosen_bitrate,
        cpu_mode=True,
        is_preview=(max_frames is not None),
    )

    swapped = skipped = 0
    tracker      = None
    tracked_bbox = None
    last_result  = None
    last_centre  = None

    detect_counter   = 0

    t_start = time.perf_counter()
    logger.info(
        "[CPU] Pipeline start: %d frames, skip=%d, detect_every=%d, height=%dp, bitrate=%s",
        total, cfg.PROCESS_EVERY_N_FRAMES, cfg.DETECT_EVERY, cfg.TARGET_HEIGHT, chosen_bitrate,
    )

    try:
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            # ── Frame skipping: reuse last result for skipped frames ───────────
            if i > 0 and (i % cfg.PROCESS_EVERY_N_FRAMES != 0):
                if last_result is not None:
                    writer.stdin.write(last_result.tobytes())
                else:
                    writer.stdin.write(frame.tobytes())
                skipped += 1
                _update_progress(db_manager, job_id, i, total,
                                 progress_start, progress_end, swapped, skipped)
                continue

            # ── Load frame ─────────────────────────────────────────────────────
            if frame is None:
                skipped += 1
                continue

            orig_h, orig_w = frame.shape[:2]

            # ── Pre-resize to TARGET_HEIGHT ────────────────────────────────────
            if cfg.TARGET_HEIGHT > 0 and orig_h > cfg.TARGET_HEIGHT:
                scale  = cfg.TARGET_HEIGHT / orig_h
                small_w = int(orig_w * scale)
                small  = cv2.resize(frame, (small_w, cfg.TARGET_HEIGHT),
                                    interpolation=cv2.INTER_LINEAR)
            else:
                scale  = 1.0
                small  = frame

            sh, sw = small.shape[:2]
            face_found = False

            # ── Face Detection / Tracking (on downscaled frame) ────────────────
            run_detection = (detect_counter % cfg.DETECT_EVERY == 0) or tracker is None

            if not run_detection and tracker is not None:
                ok, bbox_small = tracker.update(small)
                if ok:
                    tracked_bbox = tuple(int(v) for v in bbox_small)
                    face_found   = True
                else:
                    tracker       = None
                    run_detection = True

            if run_detection:
                all_faces = swapper_app.get(small)
                if all_faces:
                    all_faces.sort(key=lambda f: _bbox_area(f.bbox), reverse=True)
                    best  = all_faces[0]
                    x1s, y1s, x2s, y2s = [int(v) for v in best.bbox[:4]]
                    bw_s, bh_s = x2s - x1s, y2s - y1s
                    tracked_bbox = (x1s, y1s, bw_s, bh_s)
                    face_found   = True
                    tracker      = _make_tracker()
                    if tracker is not None:
                        tracker.init(small, tracked_bbox)
                else:
                    tracker      = None
                    tracked_bbox = None

            detect_counter += 1

            # ── Face reuse: skip inference if face barely moved ────────────────
            if face_found and tracked_bbox and last_result is not None and last_centre is not None:
                cur_centre = _centre(tracked_bbox)
                dx = abs(cur_centre[0] - last_centre[0])
                dy = abs(cur_centre[1] - last_centre[1])
                if dx < cfg.REUSE_THRESHOLD_PX and dy < cfg.REUSE_THRESHOLD_PX:
                    writer.stdin.write(last_result.tobytes())
                    skipped += 1
                    _update_progress(db_manager, job_id, i, total,
                                     progress_start, progress_end, swapped, skipped)
                    continue

            # ── Crop-based Swap (on downscaled frame) ─────────────────────────
            result_small = small.copy()

            if face_found and tracked_bbox and source_face is not None:
                x, y, bw, bh = tracked_bbox
                pad_x = int(bw * FACE_CROP_PADDING)
                pad_y = int(bh * FACE_CROP_PADDING)
                x1c = max(0, x - pad_x)
                y1c = max(0, y - pad_y)
                x2c = min(sw, x + bw + pad_x)
                y2c = min(sh, y + bh + pad_y)

                crop = small[y1c:y2c, x1c:x2c]
                crop_faces = swapper_app.get(crop)

                if crop_faces:
                    crop_faces.sort(key=lambda f: _bbox_area(f.bbox), reverse=True)
                    targets = (
                        crop_faces if face_index == -1
                        else ([crop_faces[face_index]] if face_index < len(crop_faces) else crop_faces)
                    )

                    result_crop = crop.copy()
                    did_swap    = False
                    for tf in targets:
                        try:
                            result_crop = swap_adapter.swap_face(result_crop, tf, source_face)
                            did_swap    = True
                        except Exception as e:
                            logger.debug("[CPU] Swap on crop failed: %s", e)

                    if did_swap:
                        # Direct paste — no seamlessClone in CPU mode
                        result_small = _direct_paste(small, result_crop, x1c, y1c, x2c, y2c)
                        last_centre  = _centre(tracked_bbox)
                        swapped += 1
                        
                        # Periodic validation avoids massive CPU FaceAnalysis overhead
                        val_freq = 15 if (quality and getattr(quality, "value", quality) == "fast") else 5
                        if identity_validator and (i % val_freq == 0 or i == total - 1):
                            swapped_faces = swapper_app.get(result_crop)
                            if swapped_faces:
                                swapped_faces.sort(key=lambda f: _bbox_area(f.bbox), reverse=True)
                                swapped_face = swapped_faces[0]
                                timestamp = float(i) / max(1.0, fps)
                                identity_validator.add_record(i, timestamp, source_face.embedding, swapped_face.embedding)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            else:
                skipped += 1

            # ── Scale result back to original resolution ───────────────────────
            if scale < 1.0:
                result_full = cv2.resize(result_small, (orig_w, orig_h),
                                         interpolation=cv2.INTER_LINEAR)
            else:
                result_full = result_small

            last_result = result_full

            # ── Direct Pipe Write ──────────────────────────────────────────────
            writer.stdin.write(result_full.tobytes())
            _update_progress(db_manager, job_id, i, total,
                             progress_start, progress_end, swapped, skipped)

    finally:
        cap.release()
        if writer:
            if writer.stdin:
                try:
                    writer.stdin.close()
                except Exception:
                    pass
            try:
                writer.wait(timeout=10)
            except Exception:
                writer.kill()

    elapsed = time.perf_counter() - t_start
    fps_out = total / elapsed if elapsed > 0 else 0
    logger.info(
        "[CPU] Done in %.1fs → %.1f fps | swapped=%d skipped=%d",
        elapsed, fps_out, swapped, skipped,
    )
    return swapped, skipped


# ─── Private helpers ───────────────────────────────────────────────────────────

_last_cpu_progress_time = 0.0
_last_cpu_progress_pct = -1

def _update_progress(
    db_manager, job_id, i, total,
    progress_start, progress_end, swapped, skipped
) -> None:
    global _last_cpu_progress_time, _last_cpu_progress_pct
    if db_manager is not None and job_id is not None:
        now = time.monotonic()
        span = progress_end - progress_start
        pct  = progress_start + int((i + 1) / total * span)
        # Throttle writes to at most once per 500ms or when percentage changes
        if (now - _last_cpu_progress_time >= 0.5) or (pct != _last_cpu_progress_pct) or (i == total - 1):
            _last_cpu_progress_time = now
            _last_cpu_progress_pct = pct
            db_manager.update_job(job_id, {
                "progress": pct,
                "message": f"[CPU] Frame {i+1}/{total} — swapped={swapped}, skipped={skipped}"
            })
