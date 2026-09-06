# PersonaForge AI — Local Installation Guide

This guide covers complete, step-by-step setup instructions for running PersonaForge AI locally on Windows, Linux, and macOS.

---

## 1. System Requirements

| Component | Minimum Specification | Recommended Specification |
|---|---|---|
| **Operating System** | Windows 10/11 (64-bit), Ubuntu 20.04+, macOS 12+ | Windows 11 or Ubuntu 22.04 LTS |
| **Python** | Python 3.10, 3.11, 3.12, or 3.13 | Python 3.11 (64-bit) |
| **RAM** | 8 GB | 16 GB or 32 GB |
| **GPU** | Optional (runs in multi-threaded CPU mode) | NVIDIA GPU with 6GB+ VRAM (RTX 2060+, RTX 3060+, RTX 40-series) |
| **Disk Space** | 3 GB free (models + environments) | 10 GB free SSD |
| **FFmpeg** | Required (system binary) | FFmpeg 6.0+ |

---

## 2. Operating System Setup

### A. Windows Setup (Recommended)

1. **Install Python 3.11 or 3.13**:
   Download and run the installer from [python.org](https://www.python.org/downloads/). Ensure **"Add python.exe to PATH"** is checked.

2. **Install FFmpeg**:
   Using PowerShell:
   ```powershell
   winget install Gyan.FFmpeg
   ```
   Verify installation:
   ```powershell
   ffmpeg -version
   ```

3. **Install Microsoft Visual C++ Build Tools**:
   Required by InsightFace and Cython compilation:
   Download the installer from [visualstudio.microsoft.com/visual-cpp-build-tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select **"Desktop development with C++"**.

4. **Clone the Repository**:
   ```powershell
   git clone https://github.com/himanshu-jadhav108/PersonaForge-AI.git
   cd PersonaForge-AI
   ```

5. **Create and Activate Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

6. **Install Python Packages**:
   For CPU:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   For NVIDIA GPU:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip uninstall -y onnxruntime
   pip install onnxruntime-gpu
   ```

---

### B. Linux Setup (Ubuntu / Debian)

1. **Install System Dependencies**:
   ```bash
   sudo apt-get update && sudo apt-get install -y \
       git python3 python3-pip python3-venv \
       ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
       build-essential curl
   ```

2. **Clone and Setup Virtual Environment**:
   ```bash
   git clone https://github.com/himanshu-jadhav108/PersonaForge-AI.git
   cd PersonaForge-AI
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(For NVIDIA GPU acceleration on Linux, see `docs/gpu-setup.md`)*.

---

### C. macOS Setup (Apple Silicon / Intel)

1. **Install Homebrew and FFmpeg**:
   ```bash
   brew install ffmpeg python@3.11 git
   ```

2. **Clone and Setup Virtual Environment**:
   ```bash
   git clone https://github.com/himanshu-jadhav108/PersonaForge-AI.git
   cd PersonaForge-AI
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(macOS runs the optimized multi-threaded CPU pipeline)*.

---

## 3. Verifying Model Assets

PersonaForge includes an automated model manager with checksum verification and mirror failovers:

```bash
python -c "from models.model_manager import check_models; check_models(auto_download=True)"
```

The model weights (`inswapper_128.onnx` and InsightFace `buffalo_l` assets) will download automatically into `models/` and `~/.insightface/models/`.

---

## 4. Running PersonaForge AI

Start the local server:
```bash
python main.py
```
Or with Uvicorn directly:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
**`http://127.0.0.1:8000`**

---

## 5. Running the Test Suite

Verify that all unit and integration tests pass cleanly:
```bash
pip install -r requirements-dev.txt
pytest tests/ -v -p no:cacheprovider
```
