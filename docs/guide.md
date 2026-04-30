# PersonaForge AI — Getting Started Guide

Welcome to **PersonaForge AI**, a portfolio-grade, high-fidelity local AI Face Swapping pipeline featuring Identity Consistency monitoring and an in-memory execution pipeline. 

This guide provides detailed, step-by-step instructions on setting up, running, and benchmarking the application.

---

## 📌 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Step-by-Step Installation & Setup](#2-step-by-step-installation--setup)
3. [Downloading Model Weights](#3-downloading-model-weights)
4. [Running the Application Services](#4-running-the-application-services)
5. [Configuring & Tuning the Engine](#5-configuring--tuning-the-engine)
6. [Running Technical Benchmarks](#6-running-technical-benchmarks)
7. [Common Troubleshooting & DLL Setup](#7-common-troubleshooting--dll-setup)

---

## 1. Prerequisites & System Requirements

Ensure you meet the following hardware and software bounds before setup:

### Operating System Support
* **Windows 10 / 11** (Fully Supported)
* **Linux (Ubuntu 20.04+)** (Supported)

### Hardware Requirements
* **CPU Branch**: Quad-Core Intel/AMD CPU, 16GB RAM.
* **GPU Branch**: NVIDIA GPU (GTX 1060 or better, minimum 6GB VRAM) for real-time CUDA acceleration.

### System Binaries
1. **Python 3.10.x** (Ensure Python is added to your environment `PATH`).
2. **FFmpeg**: Must be installed and added to your system `PATH`.
   - On Windows: Download from gyan.dev and add the `bin/` path to System Environment Variables.
   - Verify by running: `ffmpeg -version`
3. **NVIDIA CUDA Toolkit (11.8 or 12.x)**: Required to enable `onnxruntime-gpu` acceleration.
4. **NVIDIA cuDNN**: Installed and matching your CUDA Toolkit version.

---

## 2. Step-by-Step Installation & Setup

Follow these commands in your project terminal root to prepare the environment:

### Step 1: Create a Virtual Environment
```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate on Windows (Command Prompt)
.venv\Scripts\activate

# Activate on Windows (PowerShell)
& .venv/Scripts/activate

# Activate on Linux / macOS
source .venv/bin/activate
```

### Step 2: Install Core Python Packages
To ensure maximum library compatibility (specifically between ONNX, InsightFace, and CUDA execution providers), install dependencies using pip:

```bash
# Update pip first
python -m pip install --upgrade pip

# Install core Vision, Numerical, and AI runtime libraries
pip install opencv-python numpy pillow onnxruntime-gpu insightface

# Install backend and multipart handling
pip install fastapi uvicorn python-multipart

# Install analytics tools and dashboards
pip install streamlit plotly pandas
```

> [!NOTE]  
> If you do not have a CUDA-enabled NVIDIA GPU, install the CPU version of ONNX Runtime: `pip install onnxruntime`. The application will automatically detect this and switch to the CPU pipeline.

---

## 3. Downloading Model Weights

PersonaForge AI requires two core model sets to execute local face-swapping:

1. **InsightFace Landmark & Detection Model (`buffalo_l`)**:
   - Downloads automatically on the first run of the script and places the assets in your user home directory: `~/.insightface/models/buffalo_l/`.
   - Alternatively, manual downloads can be placed inside the `models/` folder.

2. **InSwapper Model weight (`inswapper_128.onnx`)**:
   - Download the pre-trained swapper model weight from the InsightFace open-source model releases.
   - Create a folder at `static/models/` in your workspace and place the file inside:
     `static/models/inswapper_128.onnx`

---

## 4. Running the Application Services

PersonaForge AI is split into a local API backend and a secondary dashboard.

### Service A: Core FastAPI Server
The backend controls file uploads, execution threading, validation metrics tracking, and face swap loops.

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

* **Swagger API UI Docs**: Once the server starts, navigate to `http://127.0.0.1:8000/docs` to interact directly with backend endpoints.
* **Local Web Interface**: Open the locally served HTML page `http://127.0.0.1:8000/static/index.html` (or open [static/index.html](../static/index.html)) to use the drag-and-drop web dashboard.

### Service B: Local Analytics Dashboard (Streamlit)
To visualize processed jobs, similarity charts, metrics tracking, and GPU VRAM performance telemetry:

```bash
streamlit run dashboard_app.py
```
* Access the dashboard by opening `http://localhost:8501` in your web browser.

---

## 5. Configuring & Tuning the Engine

You can tweak the performance configurations for your hardware profile by modifying the files inside the `config/` directory:

### CPU Performance Tuning (`config/config_cpu.py`)
If you run on a CPU and experience processing lag, tweak these settings to speed up processing:

```python
PROCESS_EVERY_N_FRAMES = 3   # Skips intermediate frames using KCF tracking
TARGET_HEIGHT = 720          # Forces downscaling to reduce pixels processed
DET_SIZE = (320, 320)        # Restricts the detection window bounding box
```

### GPU Performance Tuning (`config/config_gpu.py`)
Optimize VRAM allocations and quality output on NVIDIA systems:

```python
USE_SEAMLESS_CLONE = True    # Enables seamless Poisson edge blending
BITRATE = "12M"              # Video output encoding bitrate
```

---

## 6. Running Technical Benchmarks

To profile CPU vs. GPU throughput, analyze FPS performance, and plot identity consistency charts:

```bash
python -m benchmark.runner
```
The results and interactive Plotly performance graphs will be saved under the `benchmark/reports/` folder.

---

## 7. Common Troubleshooting & DLL Setup

### 1. Error: `onnxruntime_providers_shared.dll` missing
- **Cause**: Mismatched CUDA Toolkit version and ONNX Runtime version.
- **Solution**: Ensure your CUDA Toolkit and cuDNN libraries are fully registered in your Windows Environment Path variables. Download `zlibwapi.dll` if required by ONNX, and place it in the CUDA toolkit bin directory.

### 2. Error: `Microsoft Visual C++ 14.0 or greater is required`
- **Cause**: Installing `insightface` from source requires compiling C extensions.
- **Solution**: Download the *Build Tools for Visual Studio* from the Microsoft portal and check the box for "C++ build tools" during setup, then rerun the pip installer.

### 3. Error: `FFmpeg is not recognized`
- **Cause**: FFmpeg execution path is missing from your system environmental variables.
- **Solution**: Rerun environment path configuration or verify that typing `ffmpeg` in a clean command prompt successfully returns the tool's usage details.

---

## 8. Processing Modes, Modular Blending & Tracking Architecture

PersonaForge AI provides genuine differentiation between processing modes through modular blending, abstract face tracking, and dynamic resolution scaling:

### Processing Mode Differentiation Matrix

| Parameter | Fast Mode | Balanced Mode | High Mode |
|---|---|---|---|
| **Resolution Target** | Dynamic max 480p | Dynamic max 720p | Dynamic max 1080p |
| **Upscaling Rule** | Never upscales lower res | Never upscales lower res | Preserves native resolution |
| **Face Blending** | `AlphaBlend` (direct paste) | `FeatheredBlend` (Gaussian soft) | `FeatheredBlend` (or experimental clone) |
| **Detection Cadence** | Every 10 frames | Every 5 frames | Every 2 frames |
| **Validation Cadence** | Every 15 frames | Every 5 frames | Every 2 frames |
| **Target Bitrate** | 2 Mbps | 6 Mbps | 12 Mbps (or 3 Mbps CPU) |

### Modular Blending Strategies (`pipelines/blending/`)
* **`AlphaBlend`**: Fast, boundary-clipped direct paste. Ideal for CPU processing and high-throughput Fast mode with 0ms boundary calculation overhead.
* **`FeatheredBlend`**: Inscribed soft elliptical Gaussian blending. Smoothly transitions between the swapped face patch and the background video without hard rectangular seams and without Poisson discoloration artifacts.
* **`SeamlessCloneExperimental`**: Retained under the `EXPERIMENTAL` flag (Safeguard 5) for comparative benchmarking. Wraps OpenCV's Poisson equation solver with automatic fallbacks to `FeatheredBlend` on boundary error.

### Tracking Abstraction (`backend/app/tracking/`)
* **`BaseFaceTracker`**: Abstract contract (`init`, `update`, `reset`) decoupling video processing loops from specific tracking algorithms.
* **`KCFTracker`**: OpenCV-backed correlation filter tracking with bounding-box volume and aspect-ratio validation.
* **`DetectionOnlyTracker`**: Bypasses correlation filtering and signals tracking failure on every frame to mandate per-frame face detection.

---

## 9. Smart Input Analysis & Face Identity Selection

PersonaForge AI features a comprehensive pre-processing analyzer that evaluates video streams before processing, groups recurring faces into coherent identity profiles, and allows targeted face replacement:

### Video Stream Probing & Diagnostics
* **Stream Metadata**: Extracts native pixel dimensions, framerate, duration, orientation (`landscape`, `portrait`, `square`), aspect ratio, and codec.
* **Heuristic Diagnostics**:
  - **Resolution Check**: Warns if input resolution is below 480p.
  - **Illumination Telemetry**: Flags scenes with low lighting (mean luminance < 40/255) or extreme overexposure (> 225/255).
  - **Motion Blur Analysis**: Evaluates frame-level Laplacian gradient variance, warning if fast camera pan or rapid head motion causes blur.
  - **Face Stability Check**: Reports if detected faces appear only intermittently (< 20% of video duration).

### ArcFace Identity Clustering
* **Clustering Algorithm**: Extracts 512D ArcFace facial embeddings across sampled keyframes and performs cosine distance clustering ($d_{\text{cosine}} \ge 0.45$) against dynamic cluster centroids.
* **Prevalence-Ordered Person Labels**: Clusters are sorted by prevalence (detection frequency and visibility percentage), mapping primary subjects to **`Detected Person 1`**, secondary subjects to **`Detected Person 2`**, etc.
* **Optimal Frontal Thumbnail Selection**: Computes a geometric frontalness metric from 5-point landmarks (nose symmetry, eye roll alignment, mouth centering, and bounding box resolution) to select the sharpest, most frontal crop as the identity's representative thumbnail.

### REST Endpoints
* **`POST /media/analyze`**: Performs video probing, face detection, clustering, and diagnostic analysis. Accepts an uploaded video or an existing `session_id`.
* **`GET /selection/thumbnails/{filename}`**: Serves representative JPEG thumbnails for detected persons.
* **Targeted Face Swapping**: Pass `target_face_id="person_1"` to `POST /process` or `POST /preview` to swap specifically the selected person across the entire video.

