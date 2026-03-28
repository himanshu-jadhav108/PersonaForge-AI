<div align="center">
    <img src="static/assets/Persona_Forge_Ai_Banner.jpeg" width="900" alt="PersonaForge AI Banner">
  
  # 🎭 PersonaForge AI
  **The ultimate production-grade video face transformation engine.**
</div>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Development-2EA043)

## 🚀 Overview
PersonaForge AI is a high-performance video face-swapping platform designed to deliver production-grade results even on consumer-grade hardware. While most face-swapping tools require high-end GPUs to be usable, PersonaForge features a **Hardware-Aware Dual-Pipeline** that automatically adapts its processing logic to your system's capabilities.

It solves the "accessibility vs. quality" trade-off by offering a zero-compromise GPU path for final renders and a highly optimized CPU fallback for rapid prototyping and low-spec environments.

## 🎬 Demo
![PersonaForge AI Demo](./static/assets/demo.gif)
> *Realistic face transformation with temporal consistency and intelligent blending.*

---

## ✨ Features

### 🧠 Core AI Features
- **InsightFace Integration:** Leverages state-of-the-art ONNX models for high-fidelity identity transfer.
- **Hybrid Blending:** Adaptive use of **Poisson (SeamlessClone)** for GPU depth and **Alpha-Compositing** for CPU speed.
- **Identity Preservation:** Advanced normalization pass to ensure facial features remain consistent under varying lighting.

### ⚡ Performance Engineering
- **Hardware-Aware Dispatcher:** Intelligent routing that switches the entire execution logic—not just the backend—based on available CUDA providers.
- **Temporal Face Tracking:** Uses **KCF (Kernelized Correlation Filters)** to bridge frames, reducing heavy AI detection passes by up to 90%.
- **Adaptive Frame Skipping:** Intelligent 1-in-N processing for CPU previewing without losing scene context.

### 🎨 UX & Integration
- **Zero-Latency Previews:** Fast preview generator to validate swaps before committing to a full render.
- **Modular Config System:** Separate tuning constants for CPU/GPU paths to ensure stability across hardware profiles.
- **RESTful API:** Clean FastAPI backend for easy integration into existing creative workflows.

---

## 🧠 System Architecture
The system follows a modular dispatcher pattern, ensuring the "Brain" (Inference) is decoupled from the "Muscle" (Hardware Logic).

```mermaid
flowchart TD
    U1[Source Face Image]
    U2[Target Video Source]
    API[FastAPI Router]
    DET[Hardware Dispatcher<br/>CPU vs GPU Detection]
    
    subgraph GPU_Path [GPU Pipeline - High Fidelity]
        FE_G[Full Frame Extraction]
        FD_G[Face Detection]
        SW_G[Inference<br/>CUDA Accelerated]
        BL_G[Seamless Blending]
    end
    
    subgraph CPU_Path [CPU Pipeline - Optimized Fallback]
        RS_C[480p Proxy Scaling]
        SK_C[Adaptive Frame Skip]
        FD_C[Sparse Detection]
        TR_C[KCF Tracking]
        SW_C[Inference<br/>CPU Optimized]
        BL_C[Direct-Paste Blending]
    end
    
    ENC[Video Reconstruction<br/>FFmpeg NVENC/Ultrafast]
    OUT[Resulting Transformation]

    U1 --> API
    U2 --> API
    API --> DET
    DET -- "CUDA Detect" --> GPU_Path
    DET -- "CPU Only" --> CPU_Path
    GPU_Path --> ENC
    CPU_Path --> ENC
    ENC --> OUT

    classDef input fill:#E8F1FF,stroke:#2F6FED,stroke-width:2px,color:#0B1F44;
    classDef process fill:#FFF7E6,stroke:#E09F1F,stroke-width:2px,color:#4A3200;
    classDef ai fill:#EFE7FF,stroke:#7C4DFF,stroke-width:2px,color:#2B145C;
    classDef output fill:#E9FFF2,stroke:#24A148,stroke-width:2px,color:#0A3318;

    class U1,U2 input;
    class API,DET,ENC process;
    class GPU_Path,CPU_Path ai;
    class OUT output;
```

---

## 🧠 Key Engineering Decisions

### 1. The Dual-Pipeline Architecture
Instead of a simple "if/else" for CUDA, I built two distinct processing loops. The CPU path isn't just a slower version of the GPU path; it uses **proxy downscaling** and **simplified compositing** to ensure a usable 1-2 FPS experience on a laptop, where the GPU path would simply stall.

### 2. Temporal Bridges (KCF Tracking)
Running InsightFace detection on every frame is the primary bottleneck. By introducing **Kernelized Correlation Filters**, we "track" the face for 10 frames after a hit. This moves the bottleneck from CPU/GPU inference to memory-efficient tracking, enabling significant speedups.

### 3. Trade-off: Quality vs. Speed
The **preview mode** uses a 1-in-3 frame skip. While this creates a slightly jittery preview, it allows users to see the character interaction in seconds. The **final render** always defaults to 1:1 frame processing with full Poisson blending.

---

## 📊 Example Performance
*Benchmarks based on a 10-second 720p @ 30fps source video.*

| Environment | Processing Mode | Total Time | Efficiency Gain |
|---|---|---|---|
| **NVIDIA RTX 3080** | Full GPU Pipeline | ~42 sec | 1.0x (Baseline) |
| **Intel i7-11800H** | Optimized CPU (Fast) | ~115 sec | 2.7x speedup vs. Raw CPU |
| **M1 Macbook Air** | Optimized CPU (Balanced) | ~140 sec | Smooth thermal profile |

---

## ⚠️ Limitations
- **Hardware Delta:** Even with optimizations, the CPU pipeline is ~3x slower than a mid-range GPU for full-quality renders.
- **Extreme Poses:** Accuracy drops slightly during extreme side profiles or high-velocity head movements (where tracking may lose the target).
- **Lighting Inconsistency:** While blending is robust, very harsh directional lighting on the source image can sometimes create visible seams in "Direct-Paste" mode.

---

## 📁 Project Structure
 ```text
 PersonaForge/
 ├── main.py                    # FastAPI application entry point
 ├── face_swap.py               # Core face-swap engine and dispatcher
 ├── video_utils.py             # FFmpeg/CV2 media utilities
 ├── requirements.txt           # Python dependencies
 ├── environment.yml            # Conda environment (faceswap)
 ├── .gitignore
 ├── config/
 │   ├── config_cpu.py          # CPU tuning profile
 │   └── config_gpu.py          # GPU tuning profile
 ├── models/
 │   └── model_manager.py       # ONNX model validation + download manager
 ├── pipelines/
 │   ├── pipeline_cpu.py        # CPU-optimized pipeline
 │   └── pipeline_gpu.py        # GPU pipeline wrapper
 ├── scripts/
 │   ├── setup_models.py        # Model setup CLI
 │   └── check_gpu.py           # GPU diagnostics CLI
 ├── utils/
 │   └── tracker_factory.py     # Shared OpenCV tracker factory
 ├── tests/
 └── static/
 ```

---

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python 3.10+
- **Inference:** ONNX Runtime (CUDA / CPU)
- **Computer Vision:** OpenCV, InsightFace, KCF Tracking
- **Media Engine:** FFmpeg (libx264 Ultrafast / NVENC)

---

## 📦 Installation & Setup

### 1. Create and Activate Conda Environment
```bash
conda env create -f environment.yml
conda activate personaforge
```

If you prefer manual environment creation:

```bash
conda create -n personaforge python=3.10 -y
conda activate personaforge
pip install -r requirements.txt
```

### 2. Download Required Models

PersonaForge requires ONNX model files (for example `inswapper_128.onnx`) that are not included in this repository.

Use the setup command:

```bash
python scripts/setup_models.py
```

What this does:

- checks required model files
- downloads missing files automatically
- stores files under the `models/` directory

Manual fallback:

- download from https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
- place file at `models/inswapper_128.onnx`

### 3. Start the API Server
```bash
python main.py
```

Open `http://127.0.0.1:8000` in your browser.

### 4. Optional Diagnostics
```bash
python scripts/check_gpu.py
```

### 5. Quick Start (3 Commands)
```bash
conda activate personaforge
python scripts/setup_models.py
python main.py
```

---

## 🖼️ Screenshots
| Upload Interface | Real-time Preview | Final Transformation |
|---|---|---|
| ![Upload UI](./static/assets/upload-ui.png) | ![Preview UI](./static/assets/preview-ui.png) | ![Final Output](./static/assets/final-output.png) |

---

## 💡 Why I Built This
PersonaForge AI was born out of a desire to move AI research from static Jupyter notebooks into a dynamic, production-ready application. My goal was to solve the "GPU Tax"—the idea that you can't build or use high-end AI if you don't have a $1000 graphics card. By engineering the adaptive CPU fallback, I proved that smart software architecture can overcome hardware limitations.

---

## 👤 Author
**Himanshu Jadhav**  

[![GitHub](https://img.shields.io/badge/GitHub-himanshu--jadhav108-181717?style=for-the-badge&logo=github)](https://github.com/himanshu-jadhav108)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Himanshu%20Jadhav-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/himanshu-jadhav-328082339)
[![Instagram](https://img.shields.io/badge/Instagram-himanshu__jadhav__108-E4405F?style=for-the-badge&logo=instagram)](https://www.instagram.com/himanshu_jadhav_108)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20Now-F7B731?style=for-the-badge&logo=vercel)](https://himanshu-jadhav-portfolio.vercel.app/)

---
*Disclaimer: This project is intended for educational and creative use. Users must adhere to ethical standards regarding consent and privacy.*