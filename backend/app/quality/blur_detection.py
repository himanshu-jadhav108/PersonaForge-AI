import math

import cv2
import numpy as np


class LaplacianBlurDetector:
    """
    Evaluates spatial focus and blur using the Variance of Laplacian (sigma_L^2).
    Provides both legacy linear scaling and calibrated logarithmic normalization
    specified in docs/metrics.md.
    """

    SIGMA_MIN_SQ: float = 25.0  # Unusable blur baseline
    SIGMA_TARGET_SQ: float = 300.0  # Standard high-definition edge sharpness target

    @staticmethod
    def compute_variance(image: np.ndarray) -> float:
        """
        Computes the raw Variance of Laplacian across the grayscale image.
        """
        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return round(float(var), 2)

    @classmethod
    def calculate_blur_score(cls, image: np.ndarray) -> float:
        """
        Calculates blur score from 0.0 to 100.0, where higher is less blurry.
        Maintains 100% backward compatibility with FaceQualityAssessor.calculate_blur_score.
        """
        variance = cls.compute_variance(image)
        score = min(100.0, variance / 5.0)
        return round(float(score), 2)

    @classmethod
    def calculate_calibrated_sharpness(cls, image: np.ndarray) -> float:
        """
        Computes calibrated logarithmic sharpness score (0-100) per docs/metrics.md:
        NormSharp = clip((ln(1 + sigma_L^2) - ln(1 + 25.0)) / (ln(1 + 300.0) - ln(1 + 25.0)), 0, 1) * 100
        """
        variance = cls.compute_variance(image)
        if variance <= 0.0:
            return 0.0

        log_var = math.log(1.0 + variance)
        log_min = math.log(1.0 + cls.SIGMA_MIN_SQ)
        log_target = math.log(1.0 + cls.SIGMA_TARGET_SQ)

        if log_target <= log_min:
            return 0.0

        norm = (log_var - log_min) / (log_target - log_min)
        clipped = max(0.0, min(1.0, norm)) * 100.0
        return round(float(clipped), 2)
