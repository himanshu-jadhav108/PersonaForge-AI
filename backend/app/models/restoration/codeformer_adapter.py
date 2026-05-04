import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.models.restoration.base import BaseFaceRestorer

logger = logging.getLogger("personaforge.restoration.codeformer")


class CodeFormerRestorer(BaseFaceRestorer):
    """
    CodeFormer Face Restoration Adapter.
    S-Lab 1.0 Non-Commercial Research License (NTU).
    Operates strictly on 512x512 aligned facial crops.
    """

    DEFAULT_MODEL_PATH: str = "models/codeformer.onnx"

    def __init__(self, model_path: str | None = None, fidelity_weight: float = 0.7):
        self.model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        self.fidelity_weight = float(fidelity_weight)
        self.session = None
        self._initialize()

    def _initialize(self) -> None:
        if not self.model_path.exists():
            project_root = Path(__file__).resolve().parents[5]
            alt_path = project_root / self.model_path
            if alt_path.exists():
                self.model_path = alt_path

        if self.model_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=opts,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                logger.info("Loaded CodeFormer restoration model from %s", self.model_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not initialize CodeFormer session: %s", e)
                self.session = None
        else:
            logger.debug("CodeFormer model weights absent at %s", self.model_path)

    def is_available(self) -> bool:
        return self.session is not None

    def restore_crop(self, crop: np.ndarray, blend_weight: float = 1.0) -> np.ndarray:
        if crop is None or crop.size == 0 or not self.is_available():
            return crop

        orig_h, orig_w = crop.shape[:2]
        weight = max(0.0, min(1.0, float(blend_weight)))
        if weight == 0.0:
            return crop

        try:
            # 1. Resize to 512x512
            input_face = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)

            # 2. Normalize BGR -> RGB, [0, 255] -> [-1, 1], NCHW
            rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB).astype(np.float32)
            normalized = (rgb / 127.5) - 1.0
            blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

            # 3. Inference
            inputs = {self.session.get_inputs()[0].name: blob}
            if len(self.session.get_inputs()) > 1:
                # Provide fidelity weight scalar tensor if input exists
                w_tensor = np.array([self.fidelity_weight], dtype=np.float32)
                inputs[self.session.get_inputs()[1].name] = w_tensor

            outputs = self.session.run(None, inputs)
            out = outputs[0][0]

            # 4. Denormalize NCHW -> HWC
            denorm = (np.transpose(out, (1, 2, 0)) + 1.0) * 127.5
            denorm = np.clip(denorm, 0, 255).astype(np.uint8)
            restored_bgr = cv2.cvtColor(denorm, cv2.COLOR_RGB2BGR)

            if (orig_w, orig_h) != (512, 512):
                restored_bgr = cv2.resize(restored_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            if weight < 1.0:
                return cv2.addWeighted(restored_bgr, weight, crop, 1.0 - weight, 0)
            return restored_bgr

        except Exception as exc:  # noqa: BLE001
            logger.warning("CodeFormer inference error: %s. Falling back to input crop.", exc)
            return crop

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": "CodeFormer",
            "type": "AI Generative Restoration",
            "is_ai": True,
            "license": "Non-Commercial Research Only (S-Lab / NTU)",
            "model_path": str(self.model_path),
            "is_available": self.is_available(),
            "description": "Codebook lookup transformer; restricted to non-commercial research use",
        }
