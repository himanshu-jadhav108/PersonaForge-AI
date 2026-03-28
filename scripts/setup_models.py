"""CLI utility to validate and download PersonaForge model files."""

from __future__ import annotations

import argparse

from models.model_manager import MODEL_CONFIG, check_models, download_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate required PersonaForge ONNX models.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIG.keys()),
        help="Download only one model by name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download model files even if they already exist.",
    )
    args = parser.parse_args()

    print("[setup] PersonaForge model setup")

    try:
        if args.model:
            download_model(args.model, overwrite=args.force)
        elif args.force:
            for model_name in MODEL_CONFIG:
                download_model(model_name, overwrite=True)
        else:
            check_models(auto_download=True)

        print("[setup] Ready!")
        return 0
    except Exception as exc:
        print(f"[setup] Setup failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
