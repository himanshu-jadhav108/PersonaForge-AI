import logging
from typing import Any

import cv2
import numpy as np

from backend.app.quality.blur_detection import LaplacianBlurDetector
from backend.app.quality.confidence import FaceConfidenceEvaluator
from backend.app.quality.models import (
    QualityMetrics,
    QualityReport,
    TemporalStabilityMetrics,
    VideoQualityReport,
)
from backend.app.quality.sharpness import SobelSharpnessEvaluator
from backend.app.quality.stability import LandmarkStabilityEvaluator

logger = logging.getLogger("personaforge.quality")


class FaceQualityAssessor:
    """
    PersonaForge Face Quality Assessment Engine.
    Orchestrates spatial sharpness, blur variance, lighting telemetry,
    facial geometry, and temporal stability evaluations.
    """

    def __init__(self, face_analysis_app: Any = None):
        """
        :param face_analysis_app: An instance of InsightFace's FaceAnalysis app (or None).
        """
        self.app = face_analysis_app
        self.blur_detector = LaplacianBlurDetector()
        self.sharpness_evaluator = SobelSharpnessEvaluator()
        self.confidence_evaluator = FaceConfidenceEvaluator()
        self.stability_evaluator = LandmarkStabilityEvaluator()

    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        Calculates blur score based on Variance of Laplacian.
        Returns a score from 0 to 100, where higher is less blurry.
        """
        return self.blur_detector.calculate_blur_score(image)

    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculates sharpness using mean Sobel gradient magnitude.
        Returns a score from 0 to 100.
        """
        return self.sharpness_evaluator.calculate_sharpness(image)

    def calculate_lighting(self, image: np.ndarray) -> tuple[float, float]:
        """
        Returns (brightness, contrast) scores between 0 and 100.
        """
        return self.confidence_evaluator.calculate_lighting(image)

    def calculate_face_metrics(self, image: np.ndarray) -> tuple[float, float, float]:
        """
        Returns (face_angle_score, face_size_score, occlusion_score) between 0 and 100.
        """
        if not self.app:
            logger.warning("InsightFace app not provided. Returning default face metrics.")
            return 50.0, 50.0, 50.0

        faces = self.app.get(image)
        if not faces:
            return 0.0, 0.0, 0.0

        # Sort by bounding box area to get the largest face
        faces.sort(key=lambda f: self._bbox_area(f.bbox), reverse=True)
        return self.confidence_evaluator.evaluate_face_metrics(faces[0], image.shape)

    def _bbox_area(self, bbox: Any) -> float:
        """Helper for bounding box area calculation."""
        return self.confidence_evaluator.bbox_area(bbox)

    def generate_recommendations(
        self,
        blur_score: float,
        brightness: float,
        angle: float,
        occlusion: float,
        size: float,
        sharpness: float = 0.0,
    ) -> list[str]:
        """
        Generates explainable, actionable recommendations based on computed metrics.
        """
        recommendations: list[str] = []
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
            recommendations.append("Crop is too small in the frame. Position closer to the face.")

        if not recommendations:
            recommendations.append("Image quality looks great for face swapping!")

        return recommendations

    def assess_image(self, image_path: str) -> QualityReport:
        """
        Assesses the quality of a single image file on disk.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        blur_score = self.calculate_blur_score(image)
        sharpness_score = self.calculate_sharpness(image)
        brightness, contrast = self.calculate_lighting(image)
        angle, size, occlusion = self.calculate_face_metrics(image)
        raw_variance = self.blur_detector.compute_variance(image)

        lighting_score = round((brightness + contrast) / 2.0, 2)

        metrics = QualityMetrics(
            blur=blur_score,
            brightness=brightness,
            contrast=contrast,
            face_angle=angle,
            occlusion=occlusion,
            face_size=size,
            sharpness=sharpness_score,
            lighting=lighting_score,
            laplacian_variance=raw_variance,
        )

        # Overall quality score (weighted average)
        overall = (
            blur_score * 0.25
            + lighting_score * 0.15
            + angle * 0.20
            + occlusion * 0.20
            + size * 0.10
            + sharpness_score * 0.10
        )
        quality_score = round(min(100.0, max(0.0, overall)), 2)
        recommendations = self.generate_recommendations(
            blur_score=blur_score,
            brightness=brightness,
            angle=angle,
            occlusion=occlusion,
            size=size,
            sharpness=sharpness_score,
        )

        return QualityReport(
            quality_score=quality_score,
            metrics=metrics,
            recommendations=recommendations,
        )

    def assess_video_sequence(
        self,
        frames: list[np.ndarray],
        landmarks_sequence: list[np.ndarray | None] | None = None,
    ) -> VideoQualityReport:
        """
        Assesses an entire sequence of video frames, aggregating frame-level metrics
        and computing temporal landmark stability.
        """
        if not frames:
            empty_metrics = QualityMetrics(
                blur=0.0,
                brightness=0.0,
                contrast=0.0,
                face_angle=0.0,
                occlusion=0.0,
                face_size=0.0,
                sharpness=0.0,
                lighting=0.0,
            )
            return VideoQualityReport(
                overall_quality_score=0.0,
                mean_metrics=empty_metrics,
                temporal_stability=TemporalStabilityMetrics(),
                total_frames_analyzed=0,
                recommendations=["No video frames provided for assessment."],
            )

        blur_scores: list[float] = []
        sharpness_scores: list[float] = []
        brightness_scores: list[float] = []
        contrast_scores: list[float] = []
        angle_scores: list[float] = []
        size_scores: list[float] = []
        occlusion_scores: list[float] = []

        extracted_landmarks: list[np.ndarray | None] = []

        for idx, frame in enumerate(frames):
            blur_scores.append(self.calculate_blur_score(frame))
            sharpness_scores.append(self.calculate_sharpness(frame))
            b, c = self.calculate_lighting(frame)
            brightness_scores.append(b)
            contrast_scores.append(c)

            if self.app:
                faces = self.app.get(frame)
                if faces:
                    faces.sort(key=lambda f: self._bbox_area(f.bbox), reverse=True)
                    primary_face = faces[0]
                    ang, sz, occ = self.confidence_evaluator.evaluate_face_metrics(primary_face, frame.shape)
                    angle_scores.append(ang)
                    size_scores.append(sz)
                    occlusion_scores.append(occ)
                    if hasattr(primary_face, "kps"):
                        extracted_landmarks.append(primary_face.kps)
                    else:
                        extracted_landmarks.append(None)
                else:
                    angle_scores.append(0.0)
                    size_scores.append(0.0)
                    occlusion_scores.append(0.0)
                    extracted_landmarks.append(None)
            else:
                angle_scores.append(50.0)
                size_scores.append(50.0)
                occlusion_scores.append(50.0)
                if landmarks_sequence and idx < len(landmarks_sequence):
                    extracted_landmarks.append(landmarks_sequence[idx])

        # Aggregate mean values
        mean_blur = float(np.mean(blur_scores))
        mean_sharp = float(np.mean(sharpness_scores))
        mean_bright = float(np.mean(brightness_scores))
        mean_cont = float(np.mean(contrast_scores))
        mean_angle = float(np.mean(angle_scores))
        mean_size = float(np.mean(size_scores))
        mean_occ = float(np.mean(occlusion_scores))
        mean_light = round((mean_bright + mean_cont) / 2.0, 2)

        # Evaluate temporal stability
        kps_seq = landmarks_sequence if landmarks_sequence is not None else extracted_landmarks
        stability_metrics = self.stability_evaluator.evaluate_sequence(kps_seq)

        # Compute composite sequence score: 75% spatial quality + 25% temporal stability
        spatial_score = (
            mean_blur * 0.25
            + mean_light * 0.15
            + mean_angle * 0.20
            + mean_occ * 0.20
            + mean_size * 0.10
            + mean_sharp * 0.10
        )
        composite_score = (spatial_score * 0.75) + (stability_metrics.stability_score * 0.25)
        overall_score = round(min(100.0, max(0.0, composite_score)), 2)

        aggregated_metrics = QualityMetrics(
            blur=round(mean_blur, 2),
            brightness=round(mean_bright, 2),
            contrast=round(mean_cont, 2),
            face_angle=round(mean_angle, 2),
            occlusion=round(mean_occ, 2),
            face_size=round(mean_size, 2),
            sharpness=round(mean_sharp, 2),
            lighting=mean_light,
        )

        recommendations = self.generate_recommendations(
            blur_score=mean_blur,
            brightness=mean_bright,
            angle=mean_angle,
            occlusion=mean_occ,
            size=mean_size,
            sharpness=mean_sharp,
        )
        if stability_metrics.stability_score < 70.0:
            recommendations.append("Video exhibits high facial landmark jitter. Consider video stabilization.")

        return VideoQualityReport(
            overall_quality_score=overall_score,
            mean_metrics=aggregated_metrics,
            temporal_stability=stability_metrics,
            total_frames_analyzed=len(frames),
            recommendations=recommendations,
        )
