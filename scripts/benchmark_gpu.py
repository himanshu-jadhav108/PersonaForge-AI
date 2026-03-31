import os
import time
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")

def run_benchmark():
    print("\n" + "="*60)
    print("  PersonaForge AI — GPU Performance Benchmark")
    print("="*60 + "\n")

    try:
        import onnxruntime as ort
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        print("Error: insightface or onnxruntime not found. Please run with the faceswap environment.")
        return

    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    
    use_gpu = "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers
    if not use_gpu:
        print("\nWARNING: No GPU provider (CUDA/TensorRT) detected! This will run on CPU.")
    else:
        print("\nSUCCESS: GPU provider detected.")

    # 1. Load Model
    print("\n[1/3] Loading InsightFace Buffalo_L…")
    t0 = time.perf_counter()
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"])
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    print(f"      Done in {time.perf_counter() - t0:.2f}s")

    # 2. Warm up
    print("[2/3] Warming up engine…")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        app.get(dummy_frame)
    print("      Engine ready.")

    # 3. Benchmark Inference
    print("[3/3] Benchmarking 100 inferences…")
    iterations = 100
    t_start = time.perf_counter()
    for _ in range(iterations):
        app.get(dummy_frame)
    t_end = time.perf_counter()
    
    total_time = t_end - t_start
    fps = iterations / total_time
    ms_per_frame = (total_time / iterations) * 1000

    print("\n" + "-"*40)
    print(f"  RESULT:")
    print(f"  Inference Speed: {fps:.2f} FPS")
    print(f"  Latency:         {ms_per_frame:.2f} ms/frame")
    print("-"*40)

    if use_gpu and fps < 10:
        print("\nNOTE: Your GPU seems to be running slower than expected.")
        print("Check if other apps (like Ollama or Games) are using your GPU memory.")
    elif use_gpu:
        print("\nNOTE: Your GPU is performing well. If video processing is slow,")
        print("it might be due to disk I/O (frame extraction) or video encoding.")
    
    print("\nBenchmark Complete.\n")

if __name__ == "__main__":
    run_benchmark()
