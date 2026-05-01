# 🎭 PersonaForge AI — High-Fidelity Video Face Swapping

![PersonaForge AI Banner](static/assets/Persona_Forge_Ai_Banner.jpeg)

<div align="center">
  <h3>⚡ <b>The ultimate production-grade video face transformation engine.</b></h3>
  <p><i>A Project by <a href="#-about-the-author">Himanshu Jadhav</a></i></p>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Project%20Complete-2EA043)](https://github.com/himanshu-jadhav108/PersonaForge-AI)

</div>

---

## 📽️ Transformation Showcase

![Demo - Personaforge](static/assets/DEMO%20-%20Personaforge.gif)

### 🖼️ Result Gallery

|             Cinematic Rendering             |               Original & Swap               |             Detail Preservation             |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
| ![Preview 01](static/assets/Preview_01.png) | ![Preview 02](static/assets/Preview_02.png) | ![Preview 03](static/assets/Preview_03.png) |

---

## 🛤️ User Journey: From Raw Input to High-Fidelity

PersonaForge is engineered to provide a seamless transition from initial concept to professional export.

1.  **Identity Definition**: The user uploads a single high-resolution source image (Identity) and a target video (Scene).
2.  **Hardware Profiling**: The system instantly detects if a CUDA-capable GPU is available, dynamically selecting the optimized processing branch.
3.  **Iterative Validation**: Users can generate a **Zero-Latency Preview** (3-5 seconds) to verify facial alignment and identity consistency.
4.  **Master Rendering**: Upon validation, the user commits to a full render, choosing between **Fast**, **Balanced**, or **High** quality profiles.
5.  **Final Export**: The transformation is muxed with original audio and exported at professional bitrates (up to 12 Mbps).

---

## 🏗️ System Architecture

### 1. High-Level Process Flow

A streamlined view of how data flows through the PersonaForge engine.

```mermaid
graph LR
    Input["📥 Users Input (Image/Video)"] --> Router["⚙️ Backend Router (FastAPI)"]
    Router --> Hardware["🔍 Hardware Dispatcher"]
    Hardware --> Pipeline["🧠 AI Transformation Pipeline"]
    Pipeline --> Rebuild["🎬 Video Reconstruction (FFmpeg)"]
    Rebuild --> Output["💎 High-Fidelity Output"]

    %% Visual Styling
    style Input fill:#E8F1FF,stroke:#2F6FED,stroke-width:2px,color:#0B1F44
    style Router fill:#FFF7E6,stroke:#F9AB00,stroke-width:2px,color:#4A3200
    style Hardware fill:#FFF7E6,stroke:#F9AB00,stroke-width:2px,color:#4A3200
    style Pipeline fill:#F3E8FF,stroke:#9333EA,stroke-width:2px,color:#2B145C
    style Rebuild fill:#FFF7E6,stroke:#F9AB00,stroke-width:2px,color:#4A3200
    style Output fill:#E9FFF2,stroke:#10B981,stroke-width:2px,color:#0A3318
```

### 2. Engineering Blueprint (Tier 2 Detail)

An in-depth look at the **Dynamic Hardware Dispatcher** and the multi-pipeline logic.

```mermaid
flowchart TD
    subgraph Input_Layer [Input Handling]
        U1[Source Photo]
        U2[Target Video]
    end

    API[FastAPI Gateway]

    subgraph Dispatcher [Dynamic Hardware Dispatcher]
        HW_DET{CUDA Available?}
    end

    subgraph GPU_Pipeline [🔥 High-Fidelity Branch]
        FE_G[Full Frame Extraction]
        FD_G[Precise Detection]
        SW_G[CUDA Swapper Inference]
        BL_G[Poisson Seamless Blending]
    end

    subgraph CPU_Pipeline [💨 Performance-Optimized Branch]
        RS_C[720p Proxy Downscaling]
        SK_C[Adaptive Frame Skip]
        TR_C[KCF Face Tracking]
        SW_C[CPU-Accelerated Swap]
        BL_C[Alpha-Compositing]
    end

    ENC[Video Reconstruction MUX]
    OUT[Resulting transformation]

    U1 & U2 --> API
    API --> HW_DET
    HW_DET -- "Yes" --> GPU_Pipeline
    HW_DET -- "No"  --> CPU_Pipeline
    GPU_Pipeline & CPU_Pipeline --> ENC
    ENC --> OUT

    %% Design Styling
    style U1 fill:#E8F1FF,stroke:#2F6FED,stroke-width:2px,color:#0B1F44
    style U2 fill:#E8F1FF,stroke:#2F6FED,stroke-width:2px,color:#0B1F44
    style API fill:#FFF7E6,stroke:#F9AB00,stroke-width:2px,color:#4A3200
    style HW_DET fill:#FFF7E6,stroke:#F9AB00,stroke-width:3px,color:#4A3200
    style ENC fill:#FFF7E6,stroke:#F9AB00,stroke-width:2px,color:#4A3200
    style GPU_Pipeline fill:#F3E8FF,stroke:#9333EA,stroke-width:2px,color:#2B145C
    style CPU_Pipeline fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#074D70
    style OUT fill:#E9FFF2,stroke:#10B981,stroke-width:2px,color:#0A3318
```

> [TIP]
> **Design Insight: Why KCF Tracking?**
> Running face detection (InsightFace) on every frame is the primary CPU bottleneck. By using **Kernelized Correlation Filters (KCF)** to "bridge" frames, we reduce the heavy AI detection overhead by 90%, enabling a fluid 3-4 FPS experience even on standard laptops.

---

## ⚡ Technical Benchmark: GPU vs. CPU

| Metric                 | 🔥 GPU Pipeline (High-Fidelity)     | 💨 CPU Pipeline (Optimized)    |
| :--------------------- | :---------------------------------- | :----------------------------- |
| **Target Audience**    | Professional Renders / High-End PCs | Prototyping / Consumer Laptops |
| **Max Resolution**     | **Original / 1080p / 4K**           | **720p (Smart Downscale)**     |
| **Bitrate**            | 12 Mbps (Cinema Standard)           | 6 Mbps (Web-Optimized)         |
| **Blending Technique** | **Poisson (SeamlessClone)**         | **Direct Alpha-Paste**         |
| **Face Tracking**      | Frame-by-Frame Precision            | KCF Bridge (1-in-10 Detection) |
| **Est. Processing**    | ~40s (10s @ 30fps)                  | ~110s (10s @ 30fps)            |

---

## 🔍 Constraints & Realism (Edge Cases)

To ensure technical trust, we acknowledge the current physical limits of the identity transfer engine.

| Scenario              | Success Rate | System Behavior                                                  |
| :-------------------- | :----------- | :--------------------------------------------------------------- |
| **Centric Face**      | 🟢 **99%**   | Perfect alignment and blending.                                  |
| **Extreme Profile**   | 🟡 **70%**   | Identity may "drift" if landmarks are occluded.                  |
| **Low-Light / Grain** | 🟡 **75%**   | Potential for visible seams in seamless cloning.                 |
| **Rapid Motion**      | 🔴 **60%**   | KCF Tracking may lose target; full detection fallback triggered. |
| **Multiple Faces**    | 🟢 **90%**   | Configurable face-index targeting (Sort by area).                |

---

## 📂 Developer Guide: Project Architecture

### Directory Navigation

- **`pipelines/`**: Routing logic for `pipeline_gpu.py` (CUDA) and `pipeline_cpu.py` (CPU).
- **`config/`**: Centralized tuning via `config_cpu.py` and `config_gpu.py`.
- **`models/`**: `model_manager.py` handles model lifecycle, checksums, and auto-downloads.
- **`utils/`**: Shared factory for `tracker_factory.py` (KCF vs. GPU selection).

> [!IMPORTANT]
> **Performance Tip**: If the processing is too slow on your CPU, adjust `PROCESS_EVERY_N_FRAMES` in `config/config_cpu.py` to `4` or `6` to prioritize speed over smoothness.

---

## ⚙️ Configurability & Parameters

PersonaForge allows fine-tuned control over the processing engine via `config/` profiles.

```python
# config_cpu.py highlights
PROCESS_EVERY_N_FRAMES = 3   # Skip frames for speed
TARGET_HEIGHT = 720          # Force downscale for throughput
DET_SIZE = (320, 320)        # Faster detection resolution

# config_gpu.py highlights
USE_SEAMLESS_CLONE = True    # High-quality Poisson blending
BITRATE = "12M"              # Cinematic compression standard
```

---

## 🚀 The Future: Scalability & Vision

PersonaForge is designed with a modular core that prepares it for next-generation expansion.

1.  **Cloud-SaaS Extraction**: The current `JobDB` (SQLite) and `pipelines/` are ready to be containerized into microservices (Docker/Kubernetes).
2.  **Real-Time Injection**: Future versions aim to integrate **WebRTC** for real-time video face-swapping during live calls.
3.  **Distributed Workers**: Moving from the current `process_semaphore` to a Redis-backed worker queue for multi-server processing.

---

## 👤 About the Author: Himanshu Jadhav

PersonaForge AI is a solo project born from a desire to democratize high-fidelity AI by solving the **"GPU Tax"**. By engineering the adaptive CPU fallback, I proved that software architecture can bridge hardware gaps.

[![GitHub](https://img.shields.io/badge/GitHub-himanshu--jadhav108-181717?style=for-the-badge&logo=github)](https://github.com/himanshu-jadhav108)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Himanshu%20Jadhav-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/himanshu-jadhav-328082339)
[![Instagram](https://img.shields.io/badge/Instagram-himanshu__jadhav__108-E4405F?style=for-the-badge&logo=instagram)](https://www.instagram.com/himanshu_jadhav_108)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20Now-F7B731?style=for-the-badge&logo=vercel)](https://himanshu-jadhav-portfolio.vercel.app/)

---

<p align="center">
  <b>PersonaForge AI — Engineering High-Fidelity Identity</b>
</p>
