"""
artifact_detector.py — PersonaForge AI · Boundary Artifact Risk Detector

Detects boundary seam gradient discontinuities and severe color mismatches
along the swapped ROI perimeter to catch haloing, unnatural edge transitions,
and facial skin tone divergence.
"""

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field


class BoundaryArtifactReport(BaseModel):
    seam_gradient_score: float = Field(
        ..., description="Boundary gradient ratio across seam (1.0 = seamless, > 2.5 = sharp seam)"
    )
    color_discontinuity_score: float = Field(
        ..., description="CIE Lab Delta-E color distance between face perimeter and context skin"
    )
    artifact_risk_level: Literal["low", "medium", "high"] = Field(
        ..., description="Categorical boundary artifact risk rating"
    )
    coherence_score: float = Field(..., description="Calibrated composite boundary coherence score (0.0 to 100.0)")
    detected_artifacts: list[str] = Field(default_factory=list, description="List of detected boundary anomalies")
    recommendation: str = Field(..., description="Actionable recommendation to alleviate detected seam artifacts")


class BoundaryArtifactDetector:
    """
    Evaluates seamlessness, edge transition gradient continuity, and skin tone consistency
    along the perimeter of a swapped face.
    """

    SEAM_THRESHOLD_RATIO: float = 2.5
    DELTA_E_WARN_THRESHOLD: float = 12.0
    DELTA_E_HIGH_THRESHOLD: float = 24.0

    @staticmethod
    def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
        """Computes Sobel gradient magnitude of a grayscale or BGR image."""
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.float64)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

    @classmethod
    def detect_artifacts(
        cls,
        composite_frame: np.ndarray,
        bbox: tuple[int, int, int, int] | list[int] | None = None,
        mask: np.ndarray | None = None,
        band_thickness: int = 4,
    ) -> BoundaryArtifactReport:
        """
        Analyzes the boundary seam of a composite frame.

        :param composite_frame: Final blended BGR video frame.
        :param bbox: Target face bounding box [x1, y1, x2, y2].
        :param mask: Optional alpha or binary mask representing swapped region.
        :param band_thickness: Pixel width of inner and outer perimeter sampling bands.
        :return: BoundaryArtifactReport with quantitative metrics and risk classification.
        """
        if composite_frame is None or composite_frame.size == 0:
            return BoundaryArtifactReport(
                seam_gradient_score=1.0,
                color_discontinuity_score=0.0,
                artifact_risk_level="low",
                coherence_score=100.0,
                detected_artifacts=[],
                recommendation="No image data provided for boundary analysis.",
            )

        h, w = composite_frame.shape[:2]

        # Generate binary mask if bbox is provided and mask is None
        if mask is None and bbox is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
            bx1 = max(0, min(w - 1, bx1))
            by1 = max(0, min(h - 1, by1))
            bx2 = max(bx1 + 1, min(w, bx2))
            by2 = max(by1 + 1, min(h, by2))

            # Draw filled ellipse corresponding to typical face contour within bbox
            center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
            axes = (max(1, (bx2 - bx1) // 2), max(1, (by2 - by1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        elif mask is None:
            # Default to center region if neither provided
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 4, 255, -1)

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))

        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_thickness * 2 + 1, band_thickness * 2 + 1))
        dilated = cv2.dilate(binary_mask, kernel)
        eroded = cv2.erode(binary_mask, kernel)

        # Outer band (background context skin)
        outer_band = cv2.subtract(dilated, binary_mask)
        # Inner band (swapped face perimeter)
        inner_band = cv2.subtract(binary_mask, eroded)
        # Seam perimeter
        seam_perimeter = cv2.subtract(dilated, eroded)

        # 1. Gradient Continuity across Seam
        grad_mag = cls.compute_gradient_magnitude(composite_frame)
        seam_pixels = np.count_nonzero(seam_perimeter)
        inner_pixels = np.count_nonzero(eroded)

        if seam_pixels > 0:
            seam_grad = float(np.mean(grad_mag[seam_perimeter > 0]))
            context_grad = float(np.mean(grad_mag[eroded > 0])) if inner_pixels > 0 else float(np.mean(grad_mag))
            seam_ratio = 1.0 if seam_grad < 3.0 else float(seam_grad / max(3.0, context_grad))
        else:
            seam_ratio = 1.0

        # Coherence score based on seam ratio
        if seam_ratio <= 1.0:
            coherence_score = 100.0
        elif seam_ratio >= cls.SEAM_THRESHOLD_RATIO:
            coherence_score = 0.0
        else:
            fraction = (seam_ratio - 1.0) / (cls.SEAM_THRESHOLD_RATIO - 1.0)
            coherence_score = (1.0 - fraction) * 100.0

        # 2. CIE Lab Color Discontinuity (Delta-E)
        lab = cv2.cvtColor(composite_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        inner_count = np.count_nonzero(inner_band)
        outer_count = np.count_nonzero(outer_band)

        if inner_count > 0 and outer_count > 0:
            mean_lab_inner = np.array([np.mean(lab[:, :, c][inner_band > 0]) for c in range(3)], dtype=np.float32)
            mean_lab_outer = np.array([np.mean(lab[:, :, c][outer_band > 0]) for c in range(3)], dtype=np.float32)
            delta_e = float(np.linalg.norm(mean_lab_inner - mean_lab_outer))
        else:
            delta_e = 0.0

        # 3. Categorize Risk & Detected Anomalies
        detected_artifacts: list[str] = []
        if seam_ratio >= 2.0:
            detected_artifacts.append("sharp_boundary_seam")
        if delta_e >= cls.DELTA_E_HIGH_THRESHOLD:
            detected_artifacts.append("severe_color_temperature_mismatch")
        elif delta_e >= cls.DELTA_E_WARN_THRESHOLD:
            detected_artifacts.append("moderate_skin_tone_divergence")

        if delta_e < cls.DELTA_E_WARN_THRESHOLD and seam_ratio < 1.7:
            risk_level: Literal["low", "medium", "high"] = "low"
            recommendation = "Boundary seamlessly blended. Lighting and color transition are within optimal tolerances."
        elif delta_e >= cls.DELTA_E_HIGH_THRESHOLD or seam_ratio >= cls.SEAM_THRESHOLD_RATIO:
            risk_level = "high"
            recommendation = (
                "Visible boundary artifacts detected. Recommend enabling face restoration with fidelity weight "
                "w in [0.6, 0.8] and utilizing Feathered or Adaptive blending."
            )
        else:
            risk_level = "medium"
            recommendation = (
                "Mild edge transition noticeable. Consider adjusting restoration weight or selecting "
                "Feathered Blending for softer perimeter integration."
            )

        return BoundaryArtifactReport(
            seam_gradient_score=round(float(seam_ratio), 3),
            color_discontinuity_score=round(float(delta_e), 2),
            artifact_risk_level=risk_level,
            coherence_score=round(float(coherence_score), 2),
            detected_artifacts=detected_artifacts,
            recommendation=recommendation,
        )
