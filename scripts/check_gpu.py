"""GPU diagnostics helper for PersonaForge."""

from __future__ import annotations

import logging
import sys
import traceback

import onnxruntime as ort

from face_swap import FaceSwapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personaforge.gpu_check")


def check() -> int:
    print(f"Python version: {sys.version}")
    print(f"ONNX Runtime version: {ort.__version__}")

    available = ort.get_available_providers()
    print(f"Available providers: {available}")

    if "CUDAExecutionProvider" not in available:
        print("ERROR: CUDAExecutionProvider not found in available providers.")
        return 1

    print("Attempting to initialize FaceSwapper with GPU...")
    try:
        swapper = FaceSwapper(use_gpu=True)
        print("FaceSwapper initialized.")
        print(f"App providers: {swapper._app.providers}")
        return 0
    except Exception as exc:
        print(f"Initialization failed with error: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(check())
