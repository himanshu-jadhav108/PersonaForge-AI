from typing import Any

import cv2
import numpy as np

from backend.app.models.restoration.base import BaseFaceRestorer


class ClassicEnhancer(BaseFaceRestorer):
    """
    Classical image processing fallback enhancer.
    Combines bilateral edge-preserving smoothing with unsharp masking.
    Strictly reports is_ai=False per Safeguard 2 transparency requirements.
    """

    def __init__(
        self,
        bilateral_diameter: int = 7,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        unsharp_amount: float = 0.5,
    ):
        self.d = bilateral_diameter
        self.sigma_color = bilateral_sigma_color
        self.sigma_space = bilateral_sigma_space
        self.unsharp_amount = unsharp_amount

    def restore_crop(self, crop: np.ndarray, blend_weight: float = 1.0) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop

        weight = max(0.0, min(1.0, float(blend_weight)))
        if weight == 0.0:
            return crop

        # 1. Edge-preserving smoothing for facial skin
        smooth = cv2.bilateralFilter(
            crop,
            d=self.d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
        )

        # 2. Unsharp masking for eyes and facial contours
        blurred = cv2.GaussianBlur(smooth, (0, 0), sigmaX=3)
        enhanced = cv2.addWeighted(
            smooth, 1.0 + self.unsharp_amount, blurred, -self.unsharp_amount, 0
        )

        # 3. Blend with original input
        if weight < 1.0:
            return cv2.addWeighted(enhanced, weight, crop, 1.0 - weight, 0)
        return enhanced

    def is_available(self) -> bool:
        """Classical OpenCV enhancement is always available."""
        return True

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": "ClassicEnhancer",
            "type": "Classical Filter",
            "is_ai": False,
            "license": "Apache 2.0 (OpenCV Native)",
            "description": "Bilateral edge-preserving smoothing and unsharp masking fallback",
        }
