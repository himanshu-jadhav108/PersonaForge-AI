import cv2
import numpy as np


class SobelSharpnessEvaluator:
    """
    Evaluates high-frequency image edge sharpness using normalized Sobel gradient density.
    """

    @staticmethod
    def compute_sobel_gradient_magnitude(image: np.ndarray) -> np.ndarray:
        """
        Computes the gradient magnitude from horizontal and vertical Sobel filters.
        """
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.float64)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

    @classmethod
    def calculate_sharpness(cls, image: np.ndarray) -> float:
        """
        Calculates a normalized sharpness score between 0.0 and 100.0 based on mean Sobel gradient magnitude.
        Maintains backward compatibility with FaceQualityAssessor.calculate_sharpness.
        """
        if image is None or image.size == 0:
            return 0.0

        mag = cls.compute_sobel_gradient_magnitude(image)
        if mag.size == 0:
            return 0.0

        mean_mag = float(np.mean(mag))
        score = min(100.0, mean_mag * 2.0)
        return round(float(score), 2)
