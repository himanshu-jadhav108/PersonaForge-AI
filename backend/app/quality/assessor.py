import cv2
import numpy as np
import logging
from typing import Optional

from backend.app.quality.models import QualityMetrics, QualityReport

logger = logging.getLogger("personaforge.quality")

class FaceQualityAssessor:
    """
    Assesses the quality of an uploaded face image for face swapping.
    """
    def __init__(self, face_analysis_app=None):
        """
        :param face_analysis_app: An instance of InsightFace's FaceAnalysis app.
        """
        self.app = face_analysis_app

    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        Calculates blur score based on Variance of Laplacian.
        Returns a score from 0 to 100, where higher is less blurry.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Typically, variance < 100 is blurry. 
        # Map variance to 0-100 score. Assuming 1000 is very sharp.
        score = min(100.0, variance / 5.0)  # scale factor
        return round(score, 2)

    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculates sharpness. We can use a different gradient measure or reuse the blur score logic.
        Here we use Sobel derivatives for sharpness.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        mean_mag = np.mean(mag)
        
        score = min(100.0, mean_mag * 2.0)
        return round(score, 2)

    def calculate_lighting(self, image: np.ndarray) -> tuple[float, float]:
        """
        Returns (brightness, contrast) scores.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Brightness: ideal is around 128. Map to 0-100.
        # Too dark or too bright is penalized.
        diff_from_ideal = abs(mean_intensity - 128)
        brightness_score = max(0.0, 100.0 - (diff_from_ideal / 128.0) * 100.0)
        
        # Contrast: ideal standard deviation is around 40-80.
        contrast_score = min(100.0, std_intensity * 2.0)
        
        return round(brightness_score, 2), round(contrast_score, 2)

    def calculate_face_metrics(self, image: np.ndarray) -> tuple[float, float, float]:
        """
        Returns (face_angle_score, face_size_score, occlusion_score)
        """
        if not self.app:
            logger.warning("InsightFace app not provided. Returning default face metrics.")
            return 50.0, 50.0, 50.0

        faces = self.app.get(image)
        if not faces:
            return 0.0, 0.0, 0.0
            
        # Get largest face
        faces.sort(key=lambda f: self._bbox_area(f.bbox), reverse=True)
        face = faces[0]
        
        # 1. Face Size Score
        h, w = image.shape[:2]
        img_area = h * w
        face_area = self._bbox_area(face.bbox)
        # Ideal face size is maybe 10-30% of the image.
        ratio = face_area / (img_area + 1e-6)
        if ratio < 0.05:
            face_size_score = ratio * 2000.0 # scale up
        elif ratio > 0.5:
            face_size_score = max(0.0, 100.0 - (ratio - 0.5) * 100.0)
        else:
            face_size_score = 100.0
        face_size_score = min(100.0, face_size_score)
        
        # 2. Face Angle (Pose) Score
        # Pose is typically (pitch, yaw, roll)
        if hasattr(face, 'pose'):
            pitch, yaw, roll = face.pose
            # Ideal is 0, 0, 0 (frontal)
            total_dev = abs(pitch) + abs(yaw) + abs(roll)
            # typical deviation is up to 180 degrees total. We map 0 deviation to 100.
            angle_score = max(0.0, 100.0 - (total_dev / 90.0) * 100.0)
        else:
            angle_score = 80.0 # fallback

        # 3. Occlusion Score
        # Approximate using landmark detection confidence if available
        if hasattr(face, 'det_score'):
            # det_score is confidence of detection
            occlusion_score = face.det_score * 100.0
        else:
            occlusion_score = 90.0

        return round(angle_score, 2), round(face_size_score, 2), round(occlusion_score, 2)

    def _bbox_area(self, bbox) -> float:
        x1, y1, x2, y2 = bbox[:4]
        return max(0.0, float((x2 - x1) * (y2 - y1)))

    def assess_image(self, image_path: str) -> QualityReport:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        blur_score = self.calculate_blur_score(image)
        sharpness_score = self.calculate_sharpness(image)
        brightness, contrast = self.calculate_lighting(image)
        angle, size, occlusion = self.calculate_face_metrics(image)
        
        lighting_score = round((brightness + contrast) / 2.0, 2)
        
        metrics = QualityMetrics(
            blur=blur_score,
            brightness=brightness,
            contrast=contrast,
            face_angle=angle,
            occlusion=occlusion,
            face_size=size,
            sharpness=sharpness_score,
            lighting=lighting_score
        )
        
        # Overall quality score (weighted average)
        overall = (
            blur_score * 0.25 +
            lighting_score * 0.15 +
            angle * 0.20 +
            occlusion * 0.20 +
            size * 0.10 +
            sharpness_score * 0.10
        )
        quality_score = round(min(100.0, overall), 2)
        
        # Generate recommendations
        recommendations = []
        if blur_score < 60:
            recommendations.append("Image is blurry. Use a higher resolution or sharper image.")
        if brightness < 40:
            recommendations.append("Image is too dark. Use better lighting.")
        elif brightness > 90:
            recommendations.append("Image is too bright or overexposed.")
        if angle < 60:
            recommendations.append("Face is turned too much. A frontal face image works best.")
        if occlusion < 70:
            recommendations.append("Face might be partially occluded or not detected clearly.")
        if size < 40:
            recommendations.append("Face is too small in the image. Crop closer to the face.")
            
        if not recommendations:
            recommendations.append("Image quality looks great for face swapping!")
            
        return QualityReport(
            quality_score=quality_score,
            metrics=metrics,
            recommendations=recommendations
        )
