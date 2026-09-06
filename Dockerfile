# ==============================================================================
# PersonaForge AI — Production Multi-Stage Dockerfile
# Supports both CPU and NVIDIA CUDA GPU profiles
# ==============================================================================

# ------------------------------------------------------------------------------
# Base Layer: System Packages and FFmpeg
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install native multimedia, build tools, and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy runtime dependency specifications
COPY requirements.txt requirements-dev.txt ./

# ------------------------------------------------------------------------------
# Target: CPU Profile
# ------------------------------------------------------------------------------
FROM base AS cpu

# Install standard CPU runtime packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Prepare storage directories
RUN mkdir -p /app/uploads /app/outputs /app/temp_frames /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/system/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------------------
# Target: GPU Profile (NVIDIA CUDA Acceleration)
# ------------------------------------------------------------------------------
FROM base AS gpu

# Install base requirements, then install onnxruntime-gpu
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir onnxruntime-gpu>=1.18.0

# Copy application source code
COPY . /app

# Prepare storage directories
RUN mkdir -p /app/uploads /app/outputs /app/temp_frames /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/system/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
