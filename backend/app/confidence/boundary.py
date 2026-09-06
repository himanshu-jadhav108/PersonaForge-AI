import cv2
import numpy as np


class BoundaryCoherenceEvaluator:
    """
    Evaluates seamlessness and gradient continuity across facial swap boundary seams.
    Follows boundary gradient standards defined in docs/metrics.md:
    Step changes exceeding 2.5x local gradient denote unblended seams.
    """

    SEAM_THRESHOLD_RATIO: float = 2.5  # Step changes > 2.5x denote visible seam artifacts

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
    def evaluate_boundary_coherence(
        cls,
        composite_image: np.ndarray,
        mask: np.ndarray,
        dilation_radius: int = 5,
    ) -> tuple[float, float]:
        """
        Evaluates boundary gradient transition ratio across the mask edge.
        Returns:
            (boundary_ratio, coherence_score)
            - boundary_ratio: ratio of perimeter gradient to adjacent local gradient.
            - coherence_score: normalized score 0.0 to 100.0 (100 = perfectly blended).
        """
        if composite_image is None or mask is None or composite_image.size == 0 or mask.size == 0:
            return 1.0, 100.0

        h, w = composite_image.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))

        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask

        # Binarize mask
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        dilated = cv2.dilate(binary_mask, kernel)
        eroded = cv2.erode(binary_mask, kernel)

        # Boundary perimeter band B
        boundary_perimeter = cv2.subtract(dilated, eroded)
        perimeter_pixels = np.count_nonzero(boundary_perimeter)

        if perimeter_pixels == 0:
            return 1.0, 100.0

        # Interior region (local reference)
        interior_region = eroded
        interior_pixels = np.count_nonzero(interior_region)

        grad_mag = cls.compute_gradient_magnitude(composite_image)
        boundary_grad = float(np.mean(grad_mag[boundary_perimeter > 0]))

        local_grad = float(np.mean(grad_mag[interior_region > 0])) if interior_pixels > 0 else float(np.mean(grad_mag))

        ratio = boundary_grad / (local_grad + 1e-6)

        # Calibrate normalized coherence score:
        # ratio <= 1.0 -> 100%
        # ratio >= 2.5 -> 0%
        # linear interpolation in between
        if ratio <= 1.0:
            coherence_score = 100.0
        elif ratio >= cls.SEAM_THRESHOLD_RATIO:
            coherence_score = 0.0
        else:
            fraction = (ratio - 1.0) / (cls.SEAM_THRESHOLD_RATIO - 1.0)
            coherence_score = (1.0 - fraction) * 100.0

        return round(float(ratio), 3), round(float(max(0.0, min(100.0, coherence_score))), 2)
