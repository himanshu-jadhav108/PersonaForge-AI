import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("personaforge.quality.confidence")


class FaceConfidenceEvaluator:
    """
    Evaluates detection confidence, lighting quality, relative face scale,
    and head pose alignment adherence per Safeguard 7 standards.
    """

    @staticmethod
    def calculate_lighting(image: np.ndarray) -> tuple[float, float]:
        """
        Computes (brightness_score, contrast_score) between 0.0 and 100.0.
        Brightness: centered around ideal midpoint 128.
        Contrast: derived from intensity standard deviation.
        """
        if image is None or image.size == 0:
            return 0.0, 0.0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))

        # Brightness: ideal around 128; too dark or overexposed is penalized
        diff_from_ideal = abs(mean_intensity - 128.0)
        brightness_score = max(0.0, 100.0 - (diff_from_ideal / 128.0) * 100.0)

        # Contrast: ideal standard deviation ~ 40-80
        contrast_score = min(100.0, std_intensity * 2.0)

        return round(brightness_score, 2), round(contrast_score, 2)

    @staticmethod
    def bbox_area(bbox: Any) -> float:
        """
        Computes bounding box area [x1, y1, x2, y2].
        """
        if bbox is None or len(bbox) < 4:
            return 0.0
        x1, y1, x2, y2 = bbox[:4]
        return max(0.0, float((x2 - x1) * (y2 - y1)))

    @classmethod
    def calculate_face_size_score(cls, bbox: Any, image_shape: tuple[int, ...]) -> float:
        """
        Calculates face size score based on face area ratio relative to frame area.
        Optimal range is between 10% and 30% of total image area.
        """
        if bbox is None or not image_shape or len(image_shape) < 2:
            return 0.0

        h, w = image_shape[:2]
        img_area = float(h * w)
        if img_area <= 0:
            return 0.0

        face_area = cls.bbox_area(bbox)
        ratio = face_area / (img_area + 1e-6)

        if ratio < 0.05:
            score = ratio * 2000.0  # Scale up small faces
        elif ratio > 0.50:
            score = max(0.0, 100.0 - (ratio - 0.50) * 100.0)
        else:
            score = 100.0

        return round(float(min(100.0, max(0.0, score))), 2)

    @classmethod
    def calculate_face_angle_score(cls, face: Any) -> float:
        """
        Calculates face angle score based on pose or landmarks.
        Per Safeguard 7, avoids speculative deductions if uncalibrated.
        """
        if face is None:
            return 0.0

        if hasattr(face, "pose") and face.pose is not None:
            try:
                pitch, yaw, roll = face.pose
                total_dev = float(abs(pitch) + abs(yaw) + abs(roll))
                angle_score = max(0.0, 100.0 - (total_dev / 90.0) * 100.0)
                return round(float(angle_score), 2)
            except (ValueError, TypeError):
                pass

        # Fallback nominal score when pose estimation is uncalibrated
        return 80.0

    @classmethod
    def calculate_occlusion_score(cls, face: Any) -> float:
        """
        Approximates occlusion score from detection model confidence.
        """
        if face is None:
            return 0.0

        det_score = getattr(face, "det_score", None)
        if det_score is not None:
            try:
                score = float(det_score) * 100.0
                return round(float(min(100.0, max(0.0, score))), 2)
            except (ValueError, TypeError):
                pass

        return 90.0

    @classmethod
    def evaluate_face_metrics(cls, face: Any, image_shape: tuple[int, ...]) -> tuple[float, float, float]:
        """
        Evaluates (face_angle_score, face_size_score, occlusion_score) for a detected face.
        """
        if face is None:
            return 0.0, 0.0, 0.0

        bbox = getattr(face, "bbox", None)
        size_score = cls.calculate_face_size_score(bbox, image_shape)
        angle_score = cls.calculate_face_angle_score(face)
        occlusion_score = cls.calculate_occlusion_score(face)

        return angle_score, size_score, occlusion_score
