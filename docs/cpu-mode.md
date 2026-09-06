# PersonaForge AI — CPU Pipeline & Optimization Guide

PersonaForge AI is designed to run efficiently on commodity CPUs without requiring a dedicated discrete graphics card. This document explains the CPU pipeline architecture and its core optimizations.

---

## 1. Architectural Philosophy

Traditional face swapping models require significant matrix multiplication workloads during face detection, facial feature extraction, latent feature transfer, and image blending. On systems lacking CUDA acceleration, standard implementations often slow down to $< 0.5$ frames per second (FPS).

PersonaForge incorporates an optimized CPU architecture (`pipelines/pipeline_cpu.py`) that delivers interactive processing speeds ($4 - 12$ FPS on modern 6-core to 8-core CPUs) through four key innovations:

1. **Intelligent ROI Face Tracking Cache**
2. **Dynamic 720p Resolution Capping**
3. **OpenMP Multi-Threaded Inference Tuning**
4. **Vectorized NumPy / OpenCV Blending**

---

## 2. Intelligent ROI Face Tracking Cache

Full face detection with InsightFace SCRFD (`det_10g.onnx`) is the most computationally demanding step per frame.

```
Frame 0:   [SCRFD Full Detection] ──► Cache Face Bounding Box & Landmark ROI
Frame 1:   [Lightweight ROI Tracker] (Skip full detection, crop cached region)
Frame 2:   [Lightweight ROI Tracker] (Skip full detection, crop cached region)
Frame 3:   [SCRFD Full Detection] ──► Refresh & Re-align Cache
```

* **Keyframe Interval**: Full SCRFD runs once every 3 to 5 frames depending on the mode (`fast` or `balanced`).
* **Intermediate Frames**: The pipeline evaluates only the localized sub-window around the previously detected face, reducing detection compute overhead by **up to 70%**.
* **Automatic Cache Invalidation**: If landmark motion exceeds velocity thresholds or if detection confidence falls, the cache instantly invalidates and triggers a full detection pass.

---

## 3. Dynamic Resolution Downscaling

High-definition videos (1080p, 1440p, 4K) drastically increase memory and blending compute without improving the 128x128 facial latent swap fidelity:

$$\text{Target Height} = \min(\text{Original Height}, 720)$$

* In CPU mode, videos are automatically normalized to a maximum vertical resolution of **720p**.
* Bitrate is calibrated to **3M** with FFmpeg's `libx264` preset `fast`, saving massive encoding CPU cycles.

---

## 4. Multi-Threading & Thread Pool Tuning

ONNX Runtime and OpenCV leverage OpenMP and Eigen multi-threading. You can tune concurrency via environment variables:

```bash
# Recommended for 8-core CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

On Windows PowerShell:
```powershell
$env:OMP_NUM_THREADS="8"
$env:MKL_NUM_THREADS="8"
```

---

## 5. Performance Comparison

| Metric | CPU Pipeline (AMD Ryzen 7 / Intel i7) | GPU Pipeline (NVIDIA RTX 3060) |
|---|---|---|
| **Fast Mode (480p)** | 8.5 – 12.0 FPS | 45.0 – 58.0 FPS |
| **Balanced Mode (720p)** | 4.2 – 6.5 FPS | 32.0 – 42.0 FPS |
| **RAM / VRAM Usage** | ~1.4 GB System RAM | ~2.5 GB Dedicated VRAM |
| **Identity Preservation (ArcFace)** | Identical ($S \ge 0.85$) | Identical ($S \ge 0.85$) |
