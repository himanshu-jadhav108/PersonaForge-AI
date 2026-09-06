"""
seamless_clone.py — Experimental Poisson seamless clone blending.

STATUS: EXPERIMENTAL (Safeguard 5)
Retained for research and comparative benchmarking. In production, FeatheredBlend
is preferred due to deterministic boundaries and avoidance of Poisson color shifts.
"""

import logging

import cv2
import numpy as np

from pipelines.blending.base import BaseBlender
from pipelines.blending.feathered_blend import FeatheredBlend

logger = logging.getLogger("personaforge.blending.seamless_clone")


class SeamlessCloneExperimental(BaseBlender):
    """
    Experimental Poisson seamless clone blending.
    Falls back to FeatheredBlend if seamlessClone fails or coordinates exceed margins.
    """

    def __init__(self, fallback_blender: BaseBlender | None = None):
        self._fallback = fallback_blender or FeatheredBlend()

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
        crop_w = x2 - x1
        crop_h = y2 - y1

        # seamlessClone requires strict margins and valid dimensions
        if x1 < 2 or y1 < 2 or x2 > fw - 2 or y2 > fh - 2 or crop_w <= 8 or crop_h <= 8:
            return self._fallback.blend(frame, crop, x1, y1, x2, y2)

        if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        try:
            mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cx, cy = crop_w // 2, crop_h // 2
            rx = max(2, crop_w // 2 - 4)
            ry = max(2, crop_h // 2 - 4)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

            center = (x1 + cx, y1 + cy)
            return cv2.seamlessClone(crop, frame.copy(), mask, center, cv2.NORMAL_CLONE)
        except (cv2.error, RuntimeError, ValueError) as e:
            logger.debug("seamlessClone failed (%s); invoking fallback blender", e)
            return self._fallback.blend(frame, crop, x1, y1, x2, y2)
