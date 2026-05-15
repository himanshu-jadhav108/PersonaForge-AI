# PersonaForge AI — Getting Started Guide

Welcome to **PersonaForge AI**, a portfolio-grade, production-quality AI Face Swapping pipeline featuring Identity Consistency, Real-Time WebRTC streaming, and an in-memory execution pipeline.

This guide provides instructions on how to set up, run, and benchmark the application.

---

## 1. Prerequisites

Before running the application, ensure you have the following installed on your system:
* **Python 3.10+**
* **FFmpeg**: Must be installed and accessible in your system's `PATH`. (Compiled with `h264_nvenc` support if you intend to use NVIDIA GPU hardware encoding).
* **CUDA Toolkit & cuDNN**: Required for `onnxruntime-gpu` acceleration.

### Python Dependencies
Install the necessary packages using `pip`:

```bash
# Core AI and Vision
pip install insightface onnxruntime-gpu opencv-python numpy

# Backend and API
pip install fastapi uvicorn python-multipart

# Real-Time Streaming
pip install aiortc av

# Dashboard and Analytics
pip install streamlit plotly pandas
```

> [!NOTE]  
> If you do not have an NVIDIA GPU, you can install `onnxruntime` instead of `onnxruntime-gpu`. The application will automatically detect this and fallback to the CPU-optimized pipeline.

---

## 2. Running the Application Services

PersonaForge AI is split into several micro-services and dashboards. You can run them concurrently depending on your needs.

### A. Core API Server (FastAPI)
The main backend handles video processing, identity consistency, and face-swap execution. 

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
* **API Docs**: Open `http://localhost:8000/docs` in your browser to interact with the Swagger UI.
* **Uploads**: Use the `/upload` and `/process` endpoints to submit source faces and target videos.

### B. Production Analytics Dashboard (Streamlit)
The dashboard provides real-time telemetry, job queues, GPU utilization, and Identity Consistency metrics.

```bash
streamlit run dashboard_app.py
```
* **Access**: Open `http://localhost:8501` in your browser.

---

## 3. Advanced Features

### Real-Time Face Swapping (WebRTC)
To use the live webcam face-swapping feature:
1. Ensure the Core API Server is running.
2. Navigate to the `/realtime/offer` endpoint (usually integrated into the frontend UI) which negotiates the WebRTC connection.
3. The server will dynamically adjust resolution and FPS to maintain real-time performance based on your hardware capabilities.

### Running Benchmarks
To profile the system's performance (VRAM usage, FPS throughput, and Identity Drift) across different resolutions and models:

```bash
python -m benchmark.runner
```
* The benchmark runner will automatically simulate workloads and output interactive Plotly charts to the `benchmark/reports/` directory.

---

## 4. Architecture Notes

* **In-Memory Pipeline**: Video frames are streamed directly into RAM via `cv2.VideoCapture` and piped directly to an FFMPEG subprocess (`stdin`). Do not manually extract frames to disk.
* **Pluggable Models**: The system uses the Adapter Pattern (`ModelFactory`). Models like `InSwapper` or `SimSwap` can be dynamically loaded if placed in the appropriate models directory.
* **Identity Consistency**: Every processed video automatically generates a JSON report detailing the cosine similarity drift between the source face and the swapped frames over time.

---

> [!TIP]  
> For production deployments, consider running the FastAPI server using `gunicorn` with `uvicorn` workers, and put both the API and the Streamlit dashboard behind an NGINX reverse proxy.
