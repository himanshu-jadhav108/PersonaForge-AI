import json
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

import plotly.graph_objects as go

from backend.app.identity.models import FrameIdentityRecord, IdentityReport

logger = logging.getLogger("personaforge.identity")

class IdentityValidator:
    """
    Identity Consistency Engine Validator.
    Tracks similarity of swapped faces against the source face across frames.
    """

    def __init__(self, job_id: str, drift_threshold: float = 0.80):
        self.job_id = job_id
        self.drift_threshold = drift_threshold
        self.records: List[FrameIdentityRecord] = []

    def compute_similarity(self, source_emb: np.ndarray, frame_emb: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        Returns a float between -1.0 and 1.0 (typically 0.0 to 1.0 for faces).
        """
        try:
            src = np.array(source_emb, dtype=np.float32)
            tgt = np.array(frame_emb, dtype=np.float32)
            
            norm_src = np.linalg.norm(src)
            norm_tgt = np.linalg.norm(tgt)
            
            if norm_src == 0 or norm_tgt == 0:
                return 0.0
                
            score = float(np.dot(src, tgt) / (norm_src * norm_tgt))
            return score
        except Exception as e:
            logger.debug(f"Failed to compute similarity: {e}")
            return 0.0

    def detect_identity_drift(self, similarity: float) -> bool:
        """
        Detects if the identity has drifted beyond the acceptable threshold.
        """
        return similarity < self.drift_threshold

    def add_record(self, frame_index: int, timestamp: float, source_emb: np.ndarray, frame_emb: np.ndarray) -> None:
        """
        Computes similarity and adds the record to the tracked sequence.
        """
        similarity = self.compute_similarity(source_emb, frame_emb)
        is_drift = self.detect_identity_drift(similarity)
        
        record = FrameIdentityRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            similarity_score=similarity,
            is_drift=is_drift
        )
        self.records.append(record)

    def generate_identity_report(self) -> IdentityReport:
        """
        Aggregates the tracked records into a summary report.
        """
        total_frames = len(self.records)
        if total_frames == 0:
            return IdentityReport(
                job_id=self.job_id,
                identity_score=0.0,
                drift_detected=False,
                average_similarity=0.0,
                min_similarity=0.0,
                drift_occurrences=0,
                total_frames_analyzed=0,
                records=[]
            )

        similarities = [r.similarity_score for r in self.records]
        drifts = [r for r in self.records if r.is_drift]
        
        avg_sim = float(np.mean(similarities))
        min_sim = float(np.min(similarities))
        
        # Identity score from 0-100 based on average similarity (scaled)
        # Assuming typical good similarity is > 0.8, we can map 0.8 -> 80
        identity_score = max(0.0, min(100.0, avg_sim * 100))

        report = IdentityReport(
            job_id=self.job_id,
            identity_score=round(identity_score, 2),
            drift_detected=len(drifts) > 0,
            average_similarity=round(avg_sim, 3),
            min_similarity=round(min_sim, 3),
            drift_occurrences=len(drifts),
            total_frames_analyzed=total_frames,
            records=self.records
        )
        return report

    def save_report(self, output_dir: Path) -> Path:
        """
        Saves the JSON report to the specified directory.
        """
        report = self.generate_identity_report()
        report_dict = report.model_dump()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"identity_report_{self.job_id}.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
            
        logger.info(f"Identity report saved to {report_path}")
        return report_path

    def generate_visual_charts(self, output_dir: Path) -> Path:
        """
        Generates a Plotly chart showing similarity over time.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / f"identity_chart_{self.job_id}.html"
        
        if not self.records:
            logger.warning("No records to plot.")
            return chart_path
            
        frame_indices = [r.frame_index for r in self.records]
        similarities = [r.similarity_score for r in self.records]
        
        fig = go.Figure()
        
        # Line for similarity
        fig.add_trace(go.Scatter(
            x=frame_indices, 
            y=similarities,
            mode='lines+markers',
            name='Cosine Similarity',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4)
        ))
        
        # Threshold line
        fig.add_hline(
            y=self.drift_threshold, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Drift Threshold ({self.drift_threshold})", 
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title=f"Identity Consistency Over Time (Job: {self.job_id[:8]})",
            xaxis_title="Frame Index",
            yaxis_title="Cosine Similarity",
            yaxis_range=[0.0, 1.0],
            template="plotly_white"
        )
        
        fig.write_html(str(chart_path))
        logger.info(f"Identity visual chart saved to {chart_path}")
        return chart_path
