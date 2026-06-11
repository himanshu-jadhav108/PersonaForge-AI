import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
import uuid

from backend.app.selection.models import SelectionMode, FaceProfile, SelectionReport
from video_utils import get_video_info

logger = logging.getLogger("personaforge.selection")

class SmartFaceSelector:
    def __init__(self, face_analysis_app, output_dir: Path):
        self.app = face_analysis_app
        self.output_dir = output_dir
        
    def extract_faces_from_video(self, video_path: str, sample_rate_hz: float = 1.0) -> List[Dict]:
        """
        Samples the video and extracts all faces from the sampled frames.
        Returns a list of face objects with their associated frame and properties.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0
            
        frame_skip = int(fps / sample_rate_hz)
        if frame_skip < 1:
            frame_skip = 1
            
        frame_idx = 0
        extracted_faces = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip == 0:
                faces = self.app.get(frame)
                if faces:
                    for f in faces:
                        # Extract the actual crop of the face for saving later if needed
                        x1, y1, x2, y2 = [int(v) for v in f.bbox[:4]]
                        # Ensure within bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        crop = frame[y1:y2, x1:x2].copy()
                        
                        extracted_faces.append({
                            'frame_idx': frame_idx,
                            'face': f,
                            'crop': crop
                        })
            frame_idx += 1
            
        cap.release()
        return extracted_faces

    def cluster_faces(self, extracted_faces: List[Dict], threshold: float = 0.5) -> Dict[str, List[Dict]]:
        """
        Groups the extracted faces by identity using cosine similarity of their embeddings.
        Returns a dict mapping a generated cluster ID to a list of face dicts.
        """
        clusters = {}
        for item in extracted_faces:
            face = item['face']
            emb = np.array(face.embedding, dtype=np.float32)
            norm_emb = np.linalg.norm(emb)
            if norm_emb == 0:
                continue
                
            best_match_id = None
            best_sim = -1.0
            
            # Compare with existing clusters (use the first face in cluster as representative)
            for cluster_id, cluster_items in clusters.items():
                rep_emb = np.array(cluster_items[0]['face'].embedding, dtype=np.float32)
                norm_rep = np.linalg.norm(rep_emb)
                if norm_rep == 0:
                    continue
                    
                sim = float(np.dot(emb, rep_emb) / (norm_emb * norm_rep))
                if sim > best_sim:
                    best_sim = sim
                    best_match_id = cluster_id
                    
            if best_sim > threshold and best_match_id is not None:
                clusters[best_match_id].append(item)
            else:
                new_id = f"face_{uuid.uuid4().hex[:8]}"
                clusters[new_id] = [item]
                
        return clusters

    def _calculate_mouth_variance(self, cluster_items: List[Dict]) -> float:
        """
        Approximates a speaking score by calculating the variance in the distance 
        between the nose and mouth keypoints across frames in a cluster.
        In buffalo_l, kps are 5 points: [left_eye, right_eye, nose, left_mouth, right_mouth].
        """
        distances = []
        for item in cluster_items:
            face = item['face']
            if hasattr(face, 'kps') and face.kps is not None and len(face.kps) == 5:
                nose = face.kps[2]
                left_mouth = face.kps[3]
                right_mouth = face.kps[4]
                
                # Distance from nose to average mouth center
                mouth_center = (left_mouth + right_mouth) / 2.0
                dist = np.linalg.norm(nose - mouth_center)
                
                # Normalize distance by bounding box height to handle scale changes
                x1, y1, x2, y2 = face.bbox[:4]
                height = y2 - y1
                if height > 0:
                    distances.append(dist / height)
                    
        if len(distances) < 2:
            return 0.0
            
        return float(np.var(distances) * 10000.0) # Scale it up to make it readable

    def calculate_profiles(self, clusters: Dict[str, List[Dict]], total_sampled_frames: int) -> List[FaceProfile]:
        profiles = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for cluster_id, items in clusters.items():
            areas = []
            det_scores = []
            
            for item in items:
                face = item['face']
                x1, y1, x2, y2 = face.bbox[:4]
                areas.append(max(0, (x2 - x1) * (y2 - y1)))
                if hasattr(face, 'det_score'):
                    det_scores.append(face.det_score)
                    
            avg_area = float(np.mean(areas)) if areas else 0.0
            avg_det = float(np.mean(det_scores)) if det_scores else 0.0
            
            # Visibility is number of unique frames this face appears in
            unique_frames = len(set(item['frame_idx'] for item in items))
            visibility = (unique_frames / total_sampled_frames) * 100.0 if total_sampled_frames > 0 else 0.0
            
            speaking_score = self._calculate_mouth_variance(items)
            
            # Save the clearest crop as thumbnail
            items_sorted = sorted(items, key=lambda i: getattr(i['face'], 'det_score', 0.0), reverse=True)
            best_crop = items_sorted[0]['crop']
            thumbnail_name = f"{cluster_id}_thumb.jpg"
            thumbnail_path = self.output_dir / thumbnail_name
            if best_crop.size > 0:
                cv2.imwrite(str(thumbnail_path), best_crop)
            
            profile = FaceProfile(
                face_id=cluster_id,
                average_area=avg_area,
                visibility_duration=visibility,
                detection_confidence=avg_det * 100.0,
                speaking_score=speaking_score,
                thumbnail_url=f"/selection/thumbnails/{thumbnail_name}"
            )
            profiles.append(profile)
            
        return profiles

    def rank_and_select(self, profiles: List[FaceProfile], mode: SelectionMode) -> tuple[FaceProfile, float]:
        if not profiles:
            raise ValueError("No face profiles to rank.")
            
        if mode == SelectionMode.LARGEST:
            sorted_profiles = sorted(profiles, key=lambda p: p.average_area, reverse=True)
        elif mode == SelectionMode.MOST_VISIBLE or mode == SelectionMode.MOST_FREQUENT:
            sorted_profiles = sorted(profiles, key=lambda p: p.visibility_duration, reverse=True)
        elif mode == SelectionMode.MAIN_SPEAKER:
            # Combination of speaking score and visibility
            sorted_profiles = sorted(profiles, key=lambda p: (p.speaking_score * 0.7) + (p.visibility_duration * 0.3), reverse=True)
        elif mode == SelectionMode.HIGHEST_CONFIDENCE:
            sorted_profiles = sorted(profiles, key=lambda p: p.detection_confidence, reverse=True)
        else:
            sorted_profiles = sorted(profiles, key=lambda p: p.average_area, reverse=True)
            
        best = sorted_profiles[0]
        
        # Calculate a rough confidence that this is the best pick
        # If the gap between 1st and 2nd is large, confidence is high.
        if len(sorted_profiles) > 1:
            runner_up = sorted_profiles[1]
            if mode == SelectionMode.LARGEST:
                gap = (best.average_area - runner_up.average_area) / (best.average_area + 1e-6)
            elif mode == SelectionMode.MOST_VISIBLE or mode == SelectionMode.MOST_FREQUENT:
                gap = (best.visibility_duration - runner_up.visibility_duration) / 100.0
            elif mode == SelectionMode.MAIN_SPEAKER:
                gap = (best.speaking_score - runner_up.speaking_score) / (best.speaking_score + 1e-6)
            else:
                gap = (best.detection_confidence - runner_up.detection_confidence) / 100.0
            
            conf_score = min(100.0, 50.0 + (gap * 50.0))
        else:
            conf_score = 100.0
            
        return best, round(conf_score, 2)

    def analyze_video(self, job_id: str, video_path: str, mode: SelectionMode) -> SelectionReport:
        logger.info(f"Starting Smart Face Selection for {video_path} using mode {mode}")
        
        extracted_faces = self.extract_faces_from_video(video_path, sample_rate_hz=1.0)
        
        if not extracted_faces:
            raise ValueError("No faces detected in the video.")
            
        # Get total sampled frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps == 0 or np.isnan(fps): fps = 30.0
        frame_skip = int(fps / 1.0)
        if frame_skip < 1: frame_skip = 1
        total_sampled = int(frame_count / frame_skip)
        if total_sampled == 0: total_sampled = 1
        
        clusters = self.cluster_faces(extracted_faces)
        profiles = self.calculate_profiles(clusters, total_sampled)
        
        best_profile, confidence = self.rank_and_select(profiles, mode)
        
        report = SelectionReport(
            job_id=job_id,
            selected_face_id=best_profile.face_id,
            selection_mode=mode,
            confidence_score=confidence,
            profiles=profiles
        )
        
        return report
