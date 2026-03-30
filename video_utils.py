"""
video_utils.py — PersonaForge AI · Video I/O Utilities

Features:
  - Quality-aware resize: 480p / 720p / 1080p / original
  - Bitrate-aware video rebuild: 1M / 3M / 6M
  - Preview clip extraction (first N seconds)
  - GPU NVENC encoding with CPU fallback
  - Async-friendly (all blocking I/O wrapped for run_in_executor)
  - File size helper
"""

import os
import json
import shutil
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("personaforge.video_utils")


def detect_orientation(width: int, height: int) -> str:
    """Classify video orientation using dimensions."""
    if width <= 0 or height <= 0:
        return "unknown"
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _even(value: float) -> int:
    """Return a positive even integer suitable for H.264 dimensions."""
    iv = max(2, int(round(value)))
    return iv if iv % 2 == 0 else iv - 1


def compute_resize_dimensions(
    width: int,
    height: int,
    target_edge: int,
    orientation: str | None = None,
) -> tuple[int, int, bool]:
    """
    Compute resized dimensions while preserving aspect ratio.

    Orientation-aware basis:
      - landscape: scale by height
      - portrait: scale by width
      - square/unknown: scale by longest side
    """
    if width <= 0 or height <= 0 or target_edge <= 0:
        return width, height, False

    orient = orientation or detect_orientation(width, height)
    if orient == "landscape":
        basis = height
    elif orient == "portrait":
        basis = width
    else:
        basis = max(width, height)

    if basis <= target_edge:
        return width, height, False

    scale = target_edge / float(basis)
    new_w = _even(width * scale)
    new_h = _even(height * scale)
    return new_w, new_h, (new_w != width or new_h != height)


def _portrait_crop_plan(width: int, height: int) -> tuple[str, int, int] | None:
    """Build a center-crop plan to convert wider videos to 9:16 portrait."""
    if width <= 0 or height <= 0:
        return None

    target_ratio = 9.0 / 16.0
    current_ratio = width / float(height)

    # Already portrait-like; skip crop.
    if current_ratio <= target_ratio:
        return None

    crop_w = _even(height * target_ratio)
    if crop_w >= width:
        return None

    x = max(0, (width - crop_w) // 2)
    return (f"crop={crop_w}:{height}:{x}:0", crop_w, height)


# ─── FFmpeg Helper ─────────────────────────────────────────────────────────────

def _run_ffmpeg(cmd: list[str], label: str = "ffmpeg", timeout: int = 600) -> subprocess.CompletedProcess:
    logger.debug("[%s] Running: %s", label, " ".join(cmd))
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"[{label}] ffmpeg exited {result.returncode}.\n"
                f"STDERR:\n{result.stderr[-3000:]}"
            )
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[{label}] ffmpeg process timed out after {timeout}s.")


def _has_nvenc() -> bool:
    """Return True if h264_nvenc is available in this ffmpeg build."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ─── Video Info ────────────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    """Return metadata (width, height, fps, duration, total_frames, codec) for a video."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out after 30s")

    data = json.loads(result.stdout)
    info: dict = {}

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"]  = stream.get("width", 0)
            info["height"] = stream.get("height", 0)
            info["codec"]  = stream.get("codec_name", "unknown")
            info["orientation"] = detect_orientation(info["width"], info["height"])
            info["aspect_ratio"] = (
                round(info["width"] / float(info["height"]), 4)
                if info.get("height", 0)
                else 0.0
            )

            raw_fps = stream.get("avg_frame_rate", "0/1")
            try:
                num, den = raw_fps.split("/")
                info["fps"] = float(num) / float(den) if float(den) != 0 else 0.0
            except Exception:
                info["fps"] = 0.0

            dur = stream.get("duration") or data.get("format", {}).get("duration", 0)
            info["duration"]     = float(dur or 0)
            info["total_frames"] = int(stream.get("nb_frames", 0) or 0)
            break

    logger.info(
        "Video info: %dx%d @ %.2f fps, %.1fs, codec=%s",
        info.get("width", 0), info.get("height", 0),
        info.get("fps", 0),   info.get("duration", 0),
        info.get("codec", "?"),
    )
    return info


# ─── File Size Helper ──────────────────────────────────────────────────────────

def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes, or 0.0 if path doesn't exist."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


# ─── Resize ────────────────────────────────────────────────────────────────────

def resize_video(
    input_path:    str,
    output_path:   str,
    target_height: int = 720,
    resize_mode:   str = "maintain",
) -> str:
    """
    Resize/crop video with orientation-aware scaling.

    Modes:
      - maintain: preserve original framing and aspect ratio
      - crop_portrait: center-crop wider videos to 9:16 portrait, then scale

    target_height=0 means no scaling step (crop may still apply in crop_portrait mode).
    Uses ultrafast CPU encode for temporary preprocess outputs.
    """
    if resize_mode not in {"maintain", "crop_portrait"}:
        raise ValueError(f"Unsupported resize_mode '{resize_mode}'")

    info = get_video_info(input_path)
    width = int(info.get("width", 0) or 0)
    height = int(info.get("height", 0) or 0)
    orientation = info.get("orientation") or detect_orientation(width, height)

    if target_height == 0 and resize_mode == "maintain":
        shutil.copy2(input_path, output_path)
        logger.info("Resize skipped (full resolution), copied → %s", output_path)
        return output_path

    filters: list[str] = []
    work_w, work_h = width, height

    if resize_mode == "crop_portrait":
        crop_plan = _portrait_crop_plan(work_w, work_h)
        if crop_plan:
            crop_filter, crop_w, crop_h = crop_plan
            filters.append(crop_filter)
            work_w, work_h = crop_w, crop_h
            orientation = detect_orientation(work_w, work_h)

    if target_height > 0:
        new_w, new_h, resized = compute_resize_dimensions(
            work_w,
            work_h,
            target_height,
            orientation=orientation,
        )
        if resized:
            filters.append(f"scale={new_w}:{new_h}")

    if not filters:
        shutil.copy2(input_path, output_path)
        logger.info("Resize/crop not required, copied → %s", output_path)
        return output_path

    logger.info(
        "Resizing '%s' mode=%s target=%dp (%s %dx%d) …",
        input_path,
        resize_mode,
        target_height,
        orientation,
        width,
        height,
    )
    t0 = time.perf_counter()
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "copy",
        "-threads", "0",
        output_path,
    ]
    _run_ffmpeg(cmd, f"resize_{target_height}p")
    logger.info("Resize done in %.2fs → %s", time.perf_counter() - t0, output_path)
    return output_path


# ─── Frame Extraction ──────────────────────────────────────────────────────────

def extract_frames(
    video_path:  str,
    output_dir:  str,
    max_seconds: float | None = None,
) -> tuple[float, int, str | None]:
    """
    Extract frames from a video to JPEG files.

    Args:
        video_path:  Input video path.
        output_dir:  Directory to write frame_XXXXXX.jpg files.
        max_seconds: If set, extract only the first this many seconds (preview mode).

    Returns: (fps, total_frames_extracted, audio_path | None)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    fps  = info.get("fps", 30.0) or 30.0

    frame_pattern = str(out / "frame_%06d.jpg")
    logger.info("Extracting frames from '%s' @ %.2f fps …", video_path, fps)
    t0 = time.perf_counter()

    cmd = [
        "ffmpeg", "-y",
        "-threads", "0",
        "-i", video_path,
    ]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]

    cmd += [
        "-q:v", "2",
        "-f", "image2",
        frame_pattern,
    ]
    _run_ffmpeg(cmd, "extract_frames")

    frame_files  = sorted(out.glob("frame_*.jpg"))
    total_frames = len(frame_files)
    logger.info("Extracted %d frames in %.2fs → %s", total_frames, time.perf_counter() - t0, output_dir)

    if total_frames == 0:
        raise RuntimeError("Frame extraction produced 0 files. Video may be corrupt or unsupported.")

    # Extract audio (only for the full video, not previews)
    audio_path: str | None = None
    if max_seconds is None:
        audio_str = str(out.parent / f"{out.name}_audio.aac")
        try:
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn", "-acodec", "copy",
                audio_str,
            ], "extract_audio")
            audio_path = audio_str
            logger.info("Audio extracted → %s", audio_path)
        except RuntimeError:
            logger.info("No audio track found.")

    return fps, total_frames, audio_path


# ─── Video Reconstruction ──────────────────────────────────────────────────────

def rebuild_video(
    frames_dir:  str,
    audio_path:  str | None,
    output_path: str,
    fps:         float,
    bitrate:     str = "3M",
    cpu_mode:    bool = False,
) -> str:
    """
    Reconstruct an MP4 from JPEG frames.

    Args:
        frames_dir:  Directory containing frame_XXXXXX.jpg files.
        audio_path:  Optional AAC audio file to mux in.
        output_path: Output MP4 path.
        fps:         Frame rate.
        bitrate:     Target video bitrate e.g. "1M", "3M", "6M".
        cpu_mode:    If True, force libx264 ultrafast (CPU pipeline).
                     If False (default), use NVENC when available (GPU pipeline).

    Uses h264_nvenc (GPU) if available and not cpu_mode, else libx264 ultrafast/medium.
    """
    frames_path   = Path(frames_dir)
    frame_pattern = str(frames_path / "frame_%06d.jpg")
    frame_count   = len(list(frames_path.glob("frame_*.jpg")))
    if frame_count == 0:
        raise RuntimeError(f"No processed frames found in '{frames_dir}'.")

    logger.info("Rebuilding %d frames @ %.2f fps, bitrate=%s …", frame_count, fps, bitrate)
    tmp_video = output_path.replace(".mp4", "_noaudio.mp4")
    use_nvenc = (not cpu_mode) and _has_nvenc()
    t0 = time.perf_counter()

    if use_nvenc:
        logger.info("Using NVENC encoder.")
        cmd_video = [
            "ffmpeg", "-y",
            "-framerate", str(int(fps)),
            "-i", frame_pattern,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            tmp_video,
        ]
    else:
        preset = "ultrafast" if cpu_mode else "medium"
        logger.info("%s encoder (preset=%s).", "CPU libx264" if cpu_mode else "libx264 (NVENC unavail)", preset)
        cmd_video = [
            "ffmpeg", "-y",
            "-framerate", str(int(fps)),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", preset,
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-threads", "0",
            tmp_video,
        ]

    _run_ffmpeg(cmd_video, "rebuild_video")
    logger.info("Video encode done in %.2fs (bitrate=%s)", time.perf_counter() - t0, bitrate)

    # Merge audio
    if audio_path and Path(audio_path).exists():
        logger.info("Muxing audio: %s", audio_path)
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", tmp_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ], "merge_audio")
        try:
            os.remove(tmp_video)
        except OSError:
            pass
    else:
        shutil.move(tmp_video, output_path)

    logger.info("Output ready → %s", output_path)
    return output_path


# ─── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_temp_dirs(*dirs: str) -> None:
    for d in dirs:
        p = Path(d)
        if p.exists():
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                logger.debug("Cleaned: %s", d)
            except Exception as e:
                logger.warning("Failed to clean '%s': %s", d, e)
