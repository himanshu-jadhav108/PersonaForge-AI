"""
config_cpu.py — PersonaForge AI · CPU Pipeline Configuration

Applied ONLY when CUDAExecutionProvider is unavailable.
Every value here is deliberately aggressive to maximise throughput on CPU.
"""

import os

# ── Frame Skipping ────────────────────────────────────────────────────────────
# Adaptive: fewer cores → skip more frames.
_cpu_cores = os.cpu_count() or 2
if _cpu_cores <= 4:
    PROCESS_EVERY_N_FRAMES: int = 4   # Process 1 in 4 frames
else:
    PROCESS_EVERY_N_FRAMES: int = 3   # Process 1 in 3 frames

# Full face detection runs every N *processed* frames; KCF tracks between them.
DETECT_EVERY: int = 10

# ── Resolution ────────────────────────────────────────────────────────────────
# Pre-resize to this height before detection + inference (0 = no resize).
TARGET_HEIGHT: int = 480

# Detection input size passed to InsightFace prepare().
DET_SIZE: tuple[int, int] = (320, 320)

# ── Enhancement ───────────────────────────────────────────────────────────────
ENHANCEMENT_ENABLED: bool = False   # Skip GFPGAN / bilateral / unsharp

# ── Encoding ──────────────────────────────────────────────────────────────────
FFMPEG_PRESET: str = "ultrafast"
BITRATE: str = "1M"

# ── I/O Threading ────────────────────────────────────────────────────────────
JPEG_QUALITY: int = 82              # Lower quality = faster writes
WRITE_WORKERS: int = 2              # Fewer threads → less context-switch overhead

# ── Blending ──────────────────────────────────────────────────────────────────
# Seamless clone is CPU-heavy; use direct paste in CPU mode.
USE_SEAMLESS_CLONE: bool = False

# ── Face Reuse ────────────────────────────────────────────────────────────────
# If the tracked face centre moves less than this many pixels, reuse last swap.
REUSE_THRESHOLD_PX: int = 5

# ── Startup Banner ────────────────────────────────────────────────────────────
STARTUP_MSG: str = (
    "Running in CPU optimized mode (faster settings applied) — "
    f"{_cpu_cores} logical core(s) detected, "
    f"processing every {PROCESS_EVERY_N_FRAMES} frames @ {TARGET_HEIGHT}p."
)
