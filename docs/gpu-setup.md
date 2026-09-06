# PersonaForge AI — NVIDIA GPU Acceleration Setup Guide

PersonaForge AI leverages NVIDIA Tensor Cores through ONNX Runtime's `CUDAExecutionProvider` and `TensorrtExecutionProvider` to achieve real-time video face swapping speeds (25–60 FPS).

---

## 1. Prerequisites

* **NVIDIA GPU**: Maxwell architecture or newer (GeForce GTX 900+, GTX 10-series, RTX 20/30/40 series, Quadro, Tesla, A100, H100).
* **VRAM Requirements**:
  * **Fast Mode**: $\ge 2 \text{ GB}$ VRAM
  * **Balanced Mode**: $\ge 4 \text{ GB}$ VRAM
  * **High Quality Mode**: $\ge 6 \text{ GB}$ VRAM
  * **AI Face Restoration (GFPGAN / CodeFormer)**: $\ge 8 \text{ GB}$ recommended
* **NVIDIA Display Driver**: $\ge 528.33$ (Windows) or $\ge 525.60$ (Linux).

---

## 2. CUDA & cuDNN Installation

### Windows 10/11
1. Download and install **CUDA Toolkit 12.x** from the official NVIDIA Developer portal:
   [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Download **cuDNN 9.x for CUDA 12**:
   [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
3. Extract the cuDNN `bin/`, `include/`, and `lib/` files into your CUDA directory (typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`).
4. Ensure `PATH` contains:
   * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp`

### Linux (Ubuntu 22.04 LTS)
```bash
sudo apt-get install -y nvidia-cuda-toolkit
# Or install NVIDIA CUDA official repository:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 libcudnn9-cuda-12
```

---

## 3. Python Package Setup

Inside your active virtual environment:

```bash
# Remove CPU onnxruntime if previously installed
pip uninstall -y onnxruntime

# Install CUDA-enabled onnxruntime
pip install onnxruntime-gpu>=1.18.0
```

---

## 4. Hardware Verification

Run the following one-liner to verify that ONNX Runtime detects your CUDA execution provider:

```python
python -c "import onnxruntime as ort; print('Available Providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA Execution Provider not found!'"
```

If successful, the output will list:
```
Available Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 5. Execution Provider Fallback Order

PersonaForge AI initializes ONNX sessions using the following prioritized hierarchy:

1. **`CUDAExecutionProvider`**: Primary acceleration path. Automatically selects GPU `device_id=0` with `cudnn_conv_algo_search="EXHAUSTIVE"`.
2. **`TensorrtExecutionProvider`**: Optional high-throughput inference engine.
3. **`CPUExecutionProvider`**: Automatic transparent fallback if CUDA libraries or drivers are unconfigured.

The active provider is reported in the UI and via `swapper.get_execution_provider()` ("GPU" or "CPU").

---

## 6. Troubleshooting

### Problem: `ImportError: DLL load failed while importing onnxruntime_pybind11_state`
* **Cause**: Missing Microsoft Visual C++ Redistributable or mismatched CUDA DLL versions.
* **Fix**: Install the latest [Visual C++ 2015–2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and verify that `cublas64_12.dll` and `cudnn64_9.dll` reside in your system `PATH`.

### Problem: GPU Out of Memory (OOM)
* **Cause**: Processing 4K target videos or running AI restoration concurrently with other GPU workloads.
* **Fix**: Select **Fast** or **Balanced** mode, which dynamically caps resolution to 480p or 720p, reducing VRAM usage by over 60%.
