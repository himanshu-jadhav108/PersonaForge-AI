"""Utilities for validating and downloading required ONNX models."""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_CONFIG = {
    "inswapper": {
        "path": "models/inswapper_128.onnx",
        "url": "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
        "urls": [
            "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        ],
        "required": True,
    },
    "gfpgan": {
        "path": "models/gfpgan_1.4.onnx",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "urls": [
            "https://huggingface.co/antigravity/gfpgan/resolve/main/gfpgan_1.4.onnx",
        ],
        "required": False,
    },
    "codeformer": {
        "path": "models/codeformer.onnx",
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "urls": [
            "https://huggingface.co/antigravity/codeformer/resolve/main/codeformer.onnx",
        ],
        "required": False,
    },
}


def get_model_path(name: str) -> Path:
    """Return absolute path for a configured model name."""
    if name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODEL_CONFIG))}")
    return _resolve_model_path(MODEL_CONFIG[name]["path"])


def _resolve_model_path(path_value: str) -> Path:
    model_path = Path(path_value)
    if model_path.is_absolute():
        return model_path
    return PROJECT_ROOT / model_path


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _format_missing_message(model_file_name: str, model_url: str) -> str:
    return (
        f"Model '{model_file_name}' not found.\n"
        f"Please download it from: {model_url}\n"
        "and place it inside the models/ directory."
    )


def download_model(name: str, overwrite: bool = False, timeout: int = 60) -> Path:
    """Download a configured model by name and store it in the expected path, trying mirrors if necessary."""
    if name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODEL_CONFIG))}")

    cfg = MODEL_CONFIG[name]
    target_path = get_model_path(name)
    urls = cfg.get("urls", [cfg["url"]])

    _ensure_models_dir()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not overwrite:
        print(f"[model] '{name}' already present at {target_path}")
        return target_path

    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    last_error = None

    for model_url in urls:
        print(f"[model] Downloading '{name}' from {model_url}...")
        try:
            with urlopen(model_url, timeout=timeout) as response, temp_path.open("wb") as f:
                total = response.headers.get("Content-Length")
                total_bytes = int(total) if total and total.isdigit() else 0
                chunk_size = 1024 * 1024
                downloaded = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100
                        print(f"\r[model] Progress: {pct:6.2f}%", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r[model] Downloaded: {mb:7.2f} MB", end="", flush=True)

            if temp_path.exists():
                temp_path.replace(target_path)

            print("\n[model] Download complete.")
            print(f"[model] Saved to: {target_path}")
            return target_path

        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            last_error = exc
            print(f"\n[model] Warning: mirror {model_url} failed: {exc}. Trying next mirror...")

    raise RuntimeError(
        f"Failed to download model '{name}' from all available mirrors.\n"
        f"Last error: {last_error}"
    )


def check_models(auto_download: bool = False) -> None:
    """Validate that all required models exist; optionally download missing models."""
    print("[model] Checking models...")
    _ensure_models_dir()

    missing_names: list[str] = []

    for model_name, cfg in MODEL_CONFIG.items():
        model_path = get_model_path(model_name)
        is_req = cfg.get("required", True)
        if model_path.exists():
            print(f"[model] Found: {model_path.name}")
            continue

        if is_req:
            print(f"[model] Missing required: {model_path.name}")
            missing_names.append(model_name)
        else:
            print(f"[model] Optional enhancement model '{model_name}' absent (classical fallback ready).")

    if auto_download and missing_names:
        for model_name in list(missing_names):
            try:
                download_model(model_name)
                missing_names.remove(model_name)
            except RuntimeError as exc:
                print(f"[model] {exc}")

    if missing_names:
        first_name = missing_names[0]
        cfg = MODEL_CONFIG[first_name]
        model_path = _resolve_model_path(cfg["path"])
        error = _format_missing_message(model_path.name, cfg["url"])
        raise FileNotFoundError(error)

    print("[model] Ready!")
