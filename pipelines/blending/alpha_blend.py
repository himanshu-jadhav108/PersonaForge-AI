"""
alpha_blend.py — Fast direct-paste / alpha blending for CPU and Fast mode.
"""

import cv2
import numpy as np

from pipelines.blending.base import BaseBlender


class AlphaBlend(BaseBlender):
    """
    Lightweight, direct-paste blender with shape safety and boundary clipping.
    Zero-overhead blending ideal for CPU processing and high-throughput Fast mode.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def blend(
        self,
        frame: np.ndarray,
        crop: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        fh, fw = frame.shape[:2]
        # Boundary clipping
        x1_c = max(0, min(fw, x1))
        y1_c = max(0, min(fh, y1))
        x2_c = max(0, min(fw, x2))
        y2_c = max(0, min(fh, y2))

        target_w = x2_c - x1_c
        target_h = y2_c - y1_c
        if target_w <= 0 or target_h <= 0:
            return frame

        # Resize crop if necessary to match destination box
        if crop.shape[0] != target_h or crop.shape[1] != target_w:
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        result = frame.copy()
        if self.alpha >= 0.999:
            result[y1_c:y2_c, x1_c:x2_c] = crop
        else:
            bg = result[y1_c:y2_c, x1_c:x2_c].astype(np.float32)
            fg = crop.astype(np.float32)
            blended = cv2.addWeighted(fg, self.alpha, bg, 1.0 - self.alpha, 0.0)
            result[y1_c:y2_c, x1_c:x2_c] = np.clip(blended, 0, 255).astype(np.uint8)

        return result
