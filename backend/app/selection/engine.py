import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.selection.models import (
    FaceProfile,
    MediaAnalysisResponse,
    SelectionMode,
    SelectionReport,
    VideoMetadata,
)
from video_utils import get_file_size_mb, get_video_info

logger = logging.getLogger("personaforge.selection")


def _calculate_frontal_score(face: Any, bbox_area: float) -> float:
    """
    Computes a deterministic frontal pose & clarity score (0-100) using 5-point facial landmarks.
    In InsightFace buffalo_l, kps = [left_eye, right_eye, nose, left_mouth, right_mouth].
    Frontal face exhibits horizontal eye alignment, nose centering between eyes, and mouth symmetry.
    """
    det_score = float(getattr(face, "det_score", 0.85) or 0.85)
    kps = getattr(face, "kps", None)

    if kps is None or len(kps) != 5:
        # Fallback when landmarks are unavailable
        return round(min(100.0, max(0.0, det_score * 100.0)), 2)

    kps = np.asarray(kps, dtype=np.float32)
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_mouth = kps[3]
    right_mouth = kps[4]

    # Inter-ocular distance
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1e-4:
        return round(min(100.0, max(0.0, det_score * 100.0)), 2)

    # 1. Eye tilt / roll alignment (0 = flat horizontal, 1 = severely tilted)
    eye_tilt = abs(right_eye[1] - left_eye[1]) / eye_dist
    eye_alignment = max(0.0, 1.0 - min(1.0, eye_tilt * 2.0))

    # 2. Nose horizontal centering relative to eye midpoint (yaw proxy)
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    nose_offset = abs(nose[0] - eye_mid_x) / eye_dist
    nose_symmetry = max(0.0, 1.0 - min(1.0, nose_offset * 2.5))

    # 3. Mouth centering relative to eye midpoint
    mouth_mid_x = (left_mouth[0] + right_mouth[0]) / 2.0
    mouth_offset = abs(mouth_mid_x - eye_mid_x) / eye_dist
    mouth_symmetry = max(0.0, 1.0 - min(1.0, mouth_offset * 2.5))

    # 4. Face area scale bonus (larger crops preserve more detail)
    scale_factor = min(1.0, math.sqrt(max(0.0, bbox_area)) / 160.0)

    # Weighted frontal index
    frontal_geom = (0.50 * nose_symmetry) + (0.30 * eye_alignment) + (0.20 * mouth_symmetry)
    score = (0.45 * frontal_geom + 0.40 * det_score + 0.15 * scale_factor) * 100.0

    return round(float(np.clip(score, 0.0, 100.0)), 2)


class SmartFaceSelector:
    def __init__(self, face_analysis_app: Any | None, output_dir: Path):
        self.app = face_analysis_app
        self.output_dir = Path(output_dir)

    def probe_media(self, video_path: str) -> VideoMetadata:
        """
        Extracts stream metadata including duration, dimensions, framerate, orientation, and codec.
        """
        info = get_video_info(video_path)
        size_mb = get_file_size_mb(video_path)

        return VideoMetadata(
            duration=round(float(info.get("duration", 0.0) or 0.0), 2),
            fps=round(float(info.get("fps", 0.0) or 0.0), 2),
            total_frames=int(info.get("total_frames", 0) or 0),
            width=int(info.get("width", 0) or 0),
            height=int(info.get("height", 0) or 0),
            orientation=str(info.get("orientation", "unknown")),
            aspect_ratio=round(float(info.get("aspect_ratio", 0.0) or 0.0), 4),
            codec=str(info.get("codec", "unknown")),
            file_size_mb=round(size_mb, 2),
        )

    def extract_faces_from_video(
        self,
        video_path: str,
        sample_rate_hz: float = 1.0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Samples the video and extracts all faces along with frame illumination and sharpness stats.
        Returns (extracted_faces, frame_statistics).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0

        frame_skip = int(fps / sample_rate_hz)
        frame_skip = max(frame_skip, 1)

        frame_idx = 0
        extracted_faces: list[dict[str, Any]] = []
        luminances: list[float] = []
        sharpnesses: list[float] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    # Collect frame-level diagnostic telemetry
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lum = float(np.mean(gray))
                    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    luminances.append(lum)
                    sharpnesses.append(sharp)

                    if self.app is not None:
                        faces = self.app.get(frame)
                        if faces:
                            for f in faces:
                                x1, y1, x2, y2 = [int(v) for v in f.bbox[:4]]
                                # Clamp to frame boundaries
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(frame.shape[1], x2)
                                y2 = min(frame.shape[0], y2)

                                # Margin padding for natural thumbnail crop (10%)
                                pw = int((x2 - x1) * 0.10)
                                ph = int((y2 - y1) * 0.10)
                                cx1 = max(0, x1 - pw)
                                cy1 = max(0, y1 - ph)
                                cx2 = min(frame.shape[1], x2 + pw)
                                cy2 = min(frame.shape[0], y2 + ph)

                                crop = frame[cy1:cy2, cx1:cx2].copy()
                                bbox_area = max(0.0, float((x2 - x1) * (y2 - y1)))
                                frontal_score = _calculate_frontal_score(f, bbox_area)

                                extracted_faces.append(
                                    {
                                        "frame_idx": frame_idx,
                                        "face": f,
                                        "crop": crop,
                                        "bbox_area": bbox_area,
                                        "frontal_score": frontal_score,
                                    }
                                )
                frame_idx += 1
        finally:
            cap.release()

        frame_stats = {
            "avg_luminance": float(np.mean(luminances)) if luminances else 128.0,
            "avg_sharpness": float(np.mean(sharpnesses)) if sharpnesses else 100.0,
            "sampled_frames_count": len(luminances),
        }

        return extracted_faces, frame_stats

    def cluster_faces(
        self,
        extracted_faces: list[dict[str, Any]],
        threshold: float = 0.45,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Groups extracted faces by identity using ArcFace embedding cosine similarity.
        Clusters are ordered by prevalence (detection frequency & visibility duration),
        so primary subjects consistently map to 'person_1', 'person_2', etc.
        """
        clusters: list[dict[str, Any]] = []

        for item in extracted_faces:
            face = item["face"]
            emb_raw = getattr(face, "embedding", None)
            if emb_raw is None:
                continue

            emb = np.array(emb_raw, dtype=np.float32).flatten()
            norm_emb = np.linalg.norm(emb)
            if norm_emb < 1e-6:
                continue
            norm_emb_vec = emb / norm_emb

            best_match_idx: int | None = None
            best_sim = -1.0

            # Compare against running centroid of each cluster
            for c_idx, cluster in enumerate(clusters):
                centroid = cluster["centroid"]
                sim = float(np.dot(norm_emb_vec, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_match_idx = c_idx

            if best_match_idx is not None and best_sim >= threshold:
                target_cluster = clusters[best_match_idx]
                target_cluster["items"].append(item)
                target_cluster["sum_emb"] += norm_emb_vec
                norm_sum = np.linalg.norm(target_cluster["sum_emb"])
                target_cluster["centroid"] = target_cluster["sum_emb"] / (norm_sum + 1e-6)
            else:
                clusters.append(
                    {
                        "items": [item],
                        "centroid": norm_emb_vec.copy(),
                        "sum_emb": norm_emb_vec.copy(),
                    }
                )

        # Sort clusters by prevalence (number of items descending, then bbox area)
        clusters.sort(
            key=lambda c: (len(c["items"]), sum(i.get("bbox_area", 0.0) for i in c["items"])),
            reverse=True,
        )

        # Build clean person_1, person_2... mapping
        result: dict[str, list[dict[str, Any]]] = {}
        for idx, cluster in enumerate(clusters):
            person_key = f"person_{idx + 1}"
            result[person_key] = cluster["items"]

        return result

    def _calculate_mouth_variance(self, cluster_items: list[dict[str, Any]]) -> float:
        """
        Approximates a speaking score by calculating variance in the nose-to-mouth distance across frames.
        """
        distances: list[float] = []
        for item in cluster_items:
            face = item["face"]
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) == 5:
                kps = np.asarray(kps, dtype=np.float32)
                nose = kps[2]
                left_mouth = kps[3]
                right_mouth = kps[4]

                mouth_center = (left_mouth + right_mouth) / 2.0
                dist = float(np.linalg.norm(nose - mouth_center))

                bbox = getattr(face, "bbox", None)
                if bbox is not None and len(bbox) >= 4:
                    height = float(bbox[3] - bbox[1])
                    if height > 0:
                        distances.append(dist / height)

        if len(distances) < 2:
            return 0.0

        return round(float(np.var(distances) * 10000.0), 2)

    def calculate_profiles(
        self,
        clusters: dict[str, list[dict[str, Any]]],
        total_sampled_frames: int,
        prefix: str = "",
    ) -> list[FaceProfile]:
        """
        Builds FaceProfile objects for each clustered identity, selecting the highest-quality
        frontal crop as the representative thumbnail.
        """
        profiles: list[FaceProfile] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for person_idx, (cluster_id, items) in enumerate(clusters.items()):
            areas: list[float] = []
            det_scores: list[float] = []
            frontal_scores: list[float] = []
            embeddings: list[np.ndarray] = []

            for item in items:
                face = item["face"]
                areas.append(item.get("bbox_area", 0.0))

                det = float(getattr(face, "det_score", 0.85) or 0.85)
                det_scores.append(det)

                frontal = float(
                    item.get("frontal_score", 0.0) or _calculate_frontal_score(face, item.get("bbox_area", 0.0))
                )
                frontal_scores.append(frontal)

                emb_raw = getattr(face, "embedding", None)
                if emb_raw is not None:
                    emb_arr = np.array(emb_raw, dtype=np.float32).flatten()
                    n = np.linalg.norm(emb_arr)
                    if n > 1e-6:
                        embeddings.append(emb_arr / n)

            avg_area = round(float(np.mean(areas)), 2) if areas else 0.0
            avg_det = round(float(np.mean(det_scores) * 100.0), 2) if det_scores else 0.0
            avg_frontal = round(float(np.mean(frontal_scores)), 2) if frontal_scores else 0.0

            unique_frames = len({item["frame_idx"] for item in items})
            visibility = round((unique_frames / max(1, total_sampled_frames)) * 100.0, 2)
            speaking_score = self._calculate_mouth_variance(items)

            # Representative embedding (normalized centroid)
            rep_emb: list[float] | None = None
            if embeddings:
                sum_emb = np.sum(embeddings, axis=0)
                n = np.linalg.norm(sum_emb)
                if n > 1e-6:
                    rep_emb = (sum_emb / n).tolist()

            # Select the best crop: candidate with highest frontal score & resolution
            items_sorted = sorted(
                items,
                key=lambda i: (i.get("frontal_score", 0.0), getattr(i["face"], "det_score", 0.0)),
                reverse=True,
            )
            best_crop = items_sorted[0].get("crop")

            # Build clean thumbnail filename
            thumb_name = f"{prefix}_{cluster_id}_thumb.jpg" if prefix else f"{cluster_id}_thumb.jpg"
            thumb_path = self.output_dir / thumb_name

            if best_crop is not None and isinstance(best_crop, np.ndarray) and best_crop.size > 0:
                cv2.imwrite(str(thumb_path), best_crop)

            person_label = f"Detected Person {person_idx + 1}"

            profile = FaceProfile(
                face_id=cluster_id,
                person_label=person_label,
                sample_count=len(items),
                average_area=avg_area,
                visibility_duration=visibility,
                detection_confidence=avg_det,
                speaking_score=speaking_score,
                frontal_score=avg_frontal,
                thumbnail_url=f"/selection/thumbnails/{thumb_name}",
                representative_embedding=rep_emb,
            )
            profiles.append(profile)

        return profiles

    def evaluate_warnings(
        self,
        metadata: VideoMetadata,
        frame_stats: dict[str, Any],
        profiles: list[FaceProfile],
    ) -> tuple[list[str], str]:
        """
        Produces explainable heuristic diagnostic warnings and recommends the optimal processing mode.
        """
        warnings: list[str] = []

        # 1. Resolution checks
        if metadata.height > 0 and metadata.width > 0 and (metadata.height < 480 or metadata.width < 480):
            warnings.append(f"Low video resolution ({metadata.width}x{metadata.height}). Output detail may be limited.")

        # 2. Lighting checks
        avg_lum = frame_stats.get("avg_luminance", 128.0)
        if avg_lum < 40.0:
            warnings.append(
                f"Low scene illumination detected ({avg_lum:.1f}/255). Facial detection accuracy may be degraded."
            )
        elif avg_lum > 225.0:
            warnings.append(
                f"High overexposure detected ({avg_lum:.1f}/255). Highlight clipping may reduce face swap fidelity."
            )

        # 3. Motion blur check
        avg_sharp = frame_stats.get("avg_sharpness", 100.0)
        if avg_sharp < 60.0:
            warnings.append(
                f"Significant motion blur detected (sharpness index {avg_sharp:.1f}). Facial boundaries may soften during fast motion."
            )

        # 4. Face presence & stability checks
        if not profiles:
            warnings.append(
                "No stable human faces detected across sampled video frames. Check lighting, orientation, or video content."
            )
        else:
            max_vis = max(p.visibility_duration for p in profiles)
            if max_vis < 20.0:
                warnings.append(
                    f"Detected faces appear intermittently (highest visibility is only {max_vis:.1f}%). Continuous tracking may be interrupted."
                )
            max_conf = max(p.detection_confidence for p in profiles)
            if max_conf < 70.0:
                warnings.append(
                    f"Low face detection confidence ({max_conf:.1f}%). Faces may be at extreme yaw/pitch angles or partially occluded."
                )

        # Recommended processing mode
        if metadata.duration > 120.0 or (metadata.height >= 1080 and metadata.duration > 60.0) or avg_sharp < 65.0:
            recommended_mode = "fast"
        elif (
            metadata.height >= 720
            and metadata.duration <= 45.0
            and avg_sharp >= 110.0
            and 50.0 <= avg_lum <= 200.0
            and profiles
            and max(p.detection_confidence for p in profiles) >= 85.0
        ):
            recommended_mode = "high"
        else:
            recommended_mode = "balanced"

        return warnings, recommended_mode

    def rank_and_select(self, profiles: list[FaceProfile], mode: SelectionMode) -> tuple[FaceProfile, float]:
        """
        Ranks candidates under the chosen selection criteria and returns (selected_profile, confidence_score).
        """
        if not profiles:
            raise ValueError("No face profiles to rank.")

        if mode == SelectionMode.LARGEST:
            sorted_profiles = sorted(profiles, key=lambda p: p.average_area, reverse=True)
        elif mode in (SelectionMode.MOST_VISIBLE, SelectionMode.MOST_FREQUENT):
            sorted_profiles = sorted(profiles, key=lambda p: p.visibility_duration, reverse=True)
        elif mode == SelectionMode.MAIN_SPEAKER:
            sorted_profiles = sorted(
                profiles,
                key=lambda p: (p.speaking_score * 0.7) + (p.visibility_duration * 0.3),
                reverse=True,
            )
        elif mode == SelectionMode.HIGHEST_CONFIDENCE:
            sorted_profiles = sorted(
                profiles,
                key=lambda p: (p.detection_confidence * 0.6) + (p.frontal_score * 0.4),
                reverse=True,
            )
        else:
            sorted_profiles = sorted(profiles, key=lambda p: p.average_area, reverse=True)

        best = sorted_profiles[0]

        # Calculate a calibrated confidence score (50-100%) based on separation margin
        if len(sorted_profiles) > 1:
            runner_up = sorted_profiles[1]
            if mode == SelectionMode.LARGEST:
                gap = (best.average_area - runner_up.average_area) / (best.average_area + 1e-6)
            elif mode in (SelectionMode.MOST_VISIBLE, SelectionMode.MOST_FREQUENT):
                gap = (best.visibility_duration - runner_up.visibility_duration) / 100.0
            elif mode == SelectionMode.MAIN_SPEAKER:
                gap = (best.speaking_score - runner_up.speaking_score) / (best.speaking_score + 1e-6)
            else:
                gap = (best.detection_confidence - runner_up.detection_confidence) / 100.0

            conf_score = min(100.0, 50.0 + (max(0.0, gap) * 50.0))
        else:
            conf_score = 100.0

        return best, round(conf_score, 2)

    def analyze_media(
        self,
        job_id: str,
        video_path: str,
        mode: SelectionMode = SelectionMode.LARGEST,
        sample_rate_hz: float = 1.0,
        session_id: str | None = None,
    ) -> MediaAnalysisResponse:
        """
        Unified end-to-end media analysis combining stream metadata probing,
        keyframe face detection, ArcFace identity clustering, and heuristic quality diagnostics.
        """
        logger.info("Starting Media Analysis for %s (mode: %s)", video_path, mode)

        # 1. Video stream metadata
        video_metadata = self.probe_media(video_path)

        # 2. Keyframe sampling & face extraction
        extracted_faces, frame_stats = self.extract_faces_from_video(video_path, sample_rate_hz=sample_rate_hz)

        # Determine total sampled frames
        total_sampled = frame_stats.get("sampled_frames_count", 0)
        if total_sampled <= 0:
            fps = video_metadata.fps if video_metadata.fps > 0 else 30.0
            frame_skip = max(1, int(fps / sample_rate_hz))
            total_sampled = max(1, int(video_metadata.total_frames / frame_skip))

        # 3. Clustering & profile calculation
        prefix = session_id or job_id
        if extracted_faces:
            clusters = self.cluster_faces(extracted_faces)
            profiles = self.calculate_profiles(clusters, total_sampled, prefix=prefix)
        else:
            profiles = []

        # 4. Warnings and recommended processing mode
        warnings, recommended_mode = self.evaluate_warnings(video_metadata, frame_stats, profiles)

        # 5. Selection ranking
        if profiles:
            best_profile, confidence = self.rank_and_select(profiles, mode)
            selected_face_id = best_profile.face_id
            selected_person_label = best_profile.person_label
        else:
            best_profile = None
            confidence = 0.0
            selected_face_id = "none"
            selected_person_label = "No Face Detected"

        return MediaAnalysisResponse(
            job_id=job_id,
            session_id=session_id,
            video_metadata=video_metadata,
            selected_face_id=selected_face_id,
            selected_person_label=selected_person_label,
            selection_mode=mode,
            confidence_score=confidence,
            profiles=profiles,
            detected_identities=profiles,
            warnings=warnings,
            recommended_mode=recommended_mode,
        )

    def analyze_video(self, job_id: str, video_path: str, mode: SelectionMode) -> SelectionReport:
        """
        Backward-compatible wrapper for existing callers expecting SelectionReport.
        """
        return self.analyze_media(
            job_id=job_id,
            video_path=video_path,
            mode=mode,
            sample_rate_hz=1.0,
            session_id=job_id,
        )
