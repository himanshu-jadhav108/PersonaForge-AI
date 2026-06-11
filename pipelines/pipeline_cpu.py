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
  • Fewer JPEG write workers to reduce thread contention
"""

import logging
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from config import config_cpu as cfg
from utils.tracker_factory import make_tracker

logger = logging.getLogger("personaforge.pipeline_cpu")


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float((x2 - x1) * (y2 - y1)))


def _centre(bbox_xywh: tuple) -> tuple[float, float]:
    x, y, w, h = bbox_xywh
    return (x + w / 2.0, y + h / 2.0)


def _make_tracker():
    return make_tracker("CPU")


def _write_frame(path: str, frame: np.ndarray, quality: int = cfg.JPEG_QUALITY) -> None:
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _direct_paste(
    frame: np.ndarray,
    crop: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    """Paste the swapped crop back without seamlessClone (CPU mode)."""
    result = frame.copy()
    crop_h, crop_w = y2 - y1, x2 - x1
    if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
        crop = cv2.resize(crop, (crop_w, crop_h))
    result[y1:y2, x1:x2] = crop
    return result


# ─── Public API ────────────────────────────────────────────────────────────────

def process_video_cpu(
    swapper_app,            # insightface FaceAnalysis instance
    swap_adapter,           # BaseSwapModel adapter instance
    source_face,            # detected source face object
    frames_dir:  str,
    output_dir:  str,
    face_index:  int        = -1,
    max_frames:  Optional[int] = None,
    progress_start: int     = 40,
    progress_end:   int     = 80,
    db_manager              = None,
    job_id:      str        = None,
    identity_validator      = None,
) -> tuple[int, int]:
    """
    CPU-optimised face-swap loop.

    Key differences from the GPU pipeline:
      - Frames pre-scaled to 480p before any inference
      - 1-in-N frame skipping with last-result reuse
      - Detection cadence: every DETECT_EVERY *processed* frames
      - No bilateral/unsharp enhancement pass
      - Direct-paste blending (no seamlessClone)

    Returns (swapped_count, skipped_count).
    """
    FACE_CROP_PADDING = 0.30    # Slightly tighter crop for speed

    frames_path = Path(frames_dir)
    out_path    = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_path.glob("frame_*.jpg"))
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
    total = len(frame_files)
    if total == 0:
        raise RuntimeError(f"No frames found in '{frames_dir}'")

    swapped = skipped = 0
    tracker      = None
    tracked_bbox = None          # (x, y, w, h) in *original* frame coords
    last_result  = None          # reuse buffer
    last_centre  = None          # centre of tracked bbox in original frame

    detect_counter   = 0         # counts processed frames since last full detect
    write_pool       = ThreadPoolExecutor(max_workers=cfg.WRITE_WORKERS)
    write_futures    = []
    BATCH_FLUSH      = 16

    t_start = time.perf_counter()
    logger.info(
        "[CPU] Pipeline start: %d frames, skip=%d, detect_every=%d, height=%dp",
        total, cfg.PROCESS_EVERY_N_FRAMES, cfg.DETECT_EVERY, cfg.TARGET_HEIGHT,
    )

    for i, frame_file in enumerate(frame_files):

        # ── Frame skipping: reuse last result for skipped frames ───────────
        if i > 0 and (i % cfg.PROCESS_EVERY_N_FRAMES != 0):
            out_file = str(out_path / frame_file.name)
            if last_result is not None:
                write_futures.append(
                    write_pool.submit(_write_frame, out_file, last_result)
                )
            else:
                # No result yet — copy original
                raw = cv2.imread(str(frame_file))
                if raw is not None:
                    write_futures.append(
                        write_pool.submit(_write_frame, out_file, raw)
                    )
            skipped += 1
            _maybe_flush(write_futures, BATCH_FLUSH)
            _update_progress(db_manager, job_id, i, total,
                             progress_start, progress_end, swapped, skipped)
            continue

        # ── Load frame ─────────────────────────────────────────────────────
        frame = cv2.imread(str(frame_file))
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
                out_file = str(out_path / frame_file.name)
                write_futures.append(
                    write_pool.submit(_write_frame, out_file, last_result)
                )
                skipped += 1
                _maybe_flush(write_futures, BATCH_FLUSH)
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
                    
                    if identity_validator:
                        swapped_faces = swapper_app.get(result_crop)
                        if swapped_faces:
                            swapped_faces.sort(key=lambda f: _bbox_area(f.bbox), reverse=True)
                            swapped_face = swapped_faces[0]
                            identity_validator.add_record(i, float(i), source_face.embedding, swapped_face.embedding)
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

        # ── Async JPEG Write ───────────────────────────────────────────────
        out_file = str(out_path / frame_file.name)
        write_futures.append(
            write_pool.submit(_write_frame, out_file, result_full)
        )
        _maybe_flush(write_futures, BATCH_FLUSH)
        _update_progress(db_manager, job_id, i, total,
                         progress_start, progress_end, swapped, skipped)

    # Drain remaining writes
    for f in write_futures:
        try:
            f.result()
        except Exception as e:
            logger.warning("[CPU] Frame write error: %s", e)
    write_pool.shutdown(wait=False)

    elapsed = time.perf_counter() - t_start
    fps_out = total / elapsed if elapsed > 0 else 0
    logger.info(
        "[CPU] Done in %.1fs → %.1f fps | swapped=%d skipped=%d",
        elapsed, fps_out, swapped, skipped,
    )
    return swapped, skipped


# ─── Private helpers ───────────────────────────────────────────────────────────

def _maybe_flush(futures: list, limit: int) -> None:
    if len(futures) >= limit:
        for f in futures:
            f.result()
        futures.clear()


def _update_progress(
    db_manager, job_id, i, total,
    progress_start, progress_end, swapped, skipped
) -> None:
    if db_manager is not None and job_id is not None:
        span = progress_end - progress_start
        pct  = progress_start + int((i + 1) / total * span)
        db_manager.update_job(job_id, {
            "progress": pct,
            "message": f"[CPU] Frame {i+1}/{total} — swapped={swapped}, skipped={skipped}"
        })
