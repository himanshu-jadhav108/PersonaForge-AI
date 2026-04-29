"""
feathered_blend.py — Soft-edge feathered elliptical blending for Balanced and High modes.
"""

import cv2
import numpy as np

from pipelines.blending.base import BaseBlender


class FeatheredBlend(BaseBlender):
    """
    Feathered blending that smoothly transitions the boundary between the
    swapped face patch and the background frame using an elliptical Gaussian mask.
    Prevents both the harsh rectangular edges of direct paste and the Poisson color
    bleeding of seamlessClone.
    """

    def __init__(self, feather_radius: int = 11):
        # Gaussian kernel size must be odd
        self.feather_radius = feather_radius if feather_radius % 2 == 1 else feather_radius + 1

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
        x1_c = max(0, min(fw, x1))
        y1_c = max(0, min(fh, y1))
        x2_c = max(0, min(fw, x2))
        y2_c = max(0, min(fh, y2))

        crop_w = x2_c - x1_c
        crop_h = y2_c - y1_c
        if crop_w <= 4 or crop_h <= 4:
            return frame

        if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # Create soft elliptical mask
        mask = np.zeros((crop_h, crop_w), dtype=np.float32)
        cx, cy = crop_w // 2, crop_h // 2
        rx = max(2, crop_w // 2 - max(2, self.feather_radius // 2))
        ry = max(2, crop_h // 2 - max(2, self.feather_radius // 2))

        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        # Apply Gaussian blur for gradual feathering
        ksize = max(3, self.feather_radius)
        if ksize % 2 == 0:
            ksize += 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        mask = np.clip(mask, 0.0, 1.0)[..., np.newaxis]

        result = frame.copy()
        bg_roi = result[y1_c:y2_c, x1_c:x2_c].astype(np.float32)
        fg_roi = crop.astype(np.float32)

        blended_roi = fg_roi * mask + bg_roi * (1.0 - mask)
        result[y1_c:y2_c, x1_c:x2_c] = np.clip(blended_roi, 0, 255).astype(np.uint8)

        return result
