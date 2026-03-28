"""
config_gpu.py — PersonaForge AI · GPU Pipeline Configuration

Mirrors the original tuning constants exactly so that the GPU pipeline
behaviour is 100% unchanged after the CPU pipeline was added.
"""

# ── Frame Processing ──────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES: int = 1    # Process every frame (no skipping)

# Full detection every N frames; KCF tracker fills gaps (same as original).
DETECT_EVERY: int = 5              # Was DETECT_EVERY_N_FRAMES in face_swap.py

# ── Resolution ────────────────────────────────────────────────────────────────
TARGET_HEIGHT: int = 0             # 0 = keep original resolution
DET_SIZE: tuple[int, int] = (640, 640)

# ── Enhancement ───────────────────────────────────────────────────────────────
ENHANCEMENT_ENABLED: bool = True   # Bilateral / unsharp-mask quality pass

# ── Encoding ──────────────────────────────────────────────────────────────────
FFMPEG_PRESET: str = "medium"      # Used only for libx264 fallback; NVENC = p4
# Bitrate is chosen per quality level in main.py (_QUALITY_CONFIG) — unchanged.

# ── I/O Threading ────────────────────────────────────────────────────────────
JPEG_QUALITY: int = 92
WRITE_WORKERS: int = 4

# ── Blending ──────────────────────────────────────────────────────────────────
USE_SEAMLESS_CLONE: bool = True    # Full seamlessClone blending

# ── Startup Banner ────────────────────────────────────────────────────────────
STARTUP_MSG: str = "Running in GPU mode (high quality) — CUDAExecutionProvider active."
