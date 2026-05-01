import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.identity.drift_detector import IdentityDriftDetector
from backend.app.identity.extractor import FaceEmbeddingExtractor
from backend.app.identity.models import (
    FrameIdentityRecord,
    IdentityReport,
    TimelinePoint,
)
from backend.app.identity.report_generator import IdentityReportGenerator
from backend.app.identity.scoring import IdentityScorer
from backend.app.identity.timeline import IdentityTimeline

logger = logging.getLogger("personaforge.identity")


class IdentityValidator:
    """
    PersonaForge Identity Consistency Engine.
    Orchestrates embedding extraction, temporal drift monitoring across configurable empirical
    zones, rolling statistics calculation, timeline point tracking, and report generation.
    """

    def __init__(
        self,
        job_id: str,
        drift_threshold: float = 0.80,
        warning_threshold: float = 0.68,
        critical_threshold: float = 0.55,
        delta_drop_threshold: float = 0.20,
    ):
        self.job_id = job_id
        self.drift_threshold = float(drift_threshold)
        self.warning_threshold = float(warning_threshold)
        self.critical_threshold = float(critical_threshold)

        self.extractor = FaceEmbeddingExtractor()
        self.detector = IdentityDriftDetector(
            stable_threshold=self.drift_threshold,
            warning_threshold=self.warning_threshold,
            critical_threshold=self.critical_threshold,
            delta_drop_threshold=delta_drop_threshold,
            rolling_window_size=5,
        )
        self.scorer = IdentityScorer()
        self.timeline = IdentityTimeline()

        self.records: list[FrameIdentityRecord] = []
        self._similarities: list[float] = []

    def compute_similarity(self, source_emb: np.ndarray, frame_emb: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.
        Returns float between -1.0 and 1.0 (typically 0.0 to 1.0 for faces).
        """
        return self.extractor.compute_similarity(source_emb, frame_emb)

    def detect_identity_drift(self, similarity: float) -> bool:
        """
        Detects if the similarity score is below the stable drift threshold.
        """
        return self.detector.is_drift(similarity)

    def add_similarity(self, frame_index: int, timestamp: float, similarity: float) -> None:
        """
        Directly appends a pre-computed similarity score into the sequence.
        """
        sim = float(similarity)
        self._similarities.append(sim)

        # Compute trailing 5-frame rolling average
        window = self._similarities[-5:]
        rolling_avg = float(np.mean(window))

        drift_status = self.detector.evaluate(sim, rolling_avg)

        # Add to scrubbable timeline
        self.timeline.add_point(
            frame_index=frame_index,
            timestamp_sec=timestamp,
            similarity=sim,
            rolling_similarity=rolling_avg,
            drift_zone=drift_status.zone,
            is_sudden_drop=drift_status.is_sudden_drop,
        )

        # Build backward-compatible record
        record = FrameIdentityRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            similarity_score=sim,
            is_drift=drift_status.is_drift,
            rolling_similarity=round(rolling_avg, 4),
            drift_zone=drift_status.zone.value,
            sudden_drop=drift_status.is_sudden_drop,
        )
        self.records.append(record)

    def add_record(
        self,
        frame_index: int,
        timestamp: float,
        source_emb: np.ndarray,
        frame_emb: np.ndarray,
    ) -> None:
        """
        Computes similarity between source and frame embedding and appends the evaluated record.
        """
        similarity = self.compute_similarity(source_emb, frame_emb)
        self.add_similarity(frame_index, timestamp, similarity)

    def generate_identity_report(self) -> IdentityReport:
        """
        Aggregates tracked records into a standardized IdentityReport.
        """
        total_frames = len(self.records)
        if total_frames == 0:
            return IdentityReport(
                job_id=self.job_id,
                identity_score=0.0,
                drift_detected=False,
                average_similarity=0.0,
                min_similarity=0.0,
                max_similarity=0.0,
                drift_occurrences=0,
                total_frames_analyzed=0,
                stable_frame_count=0,
                warning_frame_count=0,
                critical_frame_count=0,
                sudden_drops_count=0,
                records=[],
                timeline=[],
            )

        similarities = [r.similarity_score for r in self.records]
        stats = self.scorer.compute_statistics(similarities)

        stable_count = sum(1 for r in self.records if r.drift_zone == "stable")
        warning_count = sum(1 for r in self.records if r.drift_zone == "warning")
        critical_count = sum(1 for r in self.records if r.drift_zone == "critical")
        sudden_drops = sum(1 for r in self.records if getattr(r, 'sudden_drop', False))
        drifts = [r for r in self.records if r.is_drift]

        identity_score = self.scorer.calculate_identity_score(
            similarities,
            critical_count=critical_count,
            sudden_drops=sudden_drops,
        )

        return IdentityReport(
            job_id=self.job_id,
            identity_score=round(identity_score, 2),
            drift_detected=len(drifts) > 0,
            average_similarity=round(stats["mean"], 3),
            min_similarity=round(stats["min"], 3),
            max_similarity=round(stats["max"], 3),
            drift_occurrences=len(drifts),
            total_frames_analyzed=total_frames,
            stable_frame_count=stable_count,
            warning_frame_count=warning_count,
            critical_frame_count=critical_count,
            sudden_drops_count=sudden_drops,
            records=self.records,
            timeline=self.timeline.get_points(),
        )

    def save_report(self, output_dir: Path) -> Path:
        """
        Saves structured JSON report to disk and updates report with path.
        """
        report = self.generate_identity_report()
        report_path = IdentityReportGenerator.save_json_report(report, output_dir)
        report.json_report_path = str(report_path)
        return report_path

    def generate_visual_charts(self, output_dir: Path) -> Path:
        """
        Generates interactive Plotly visual chart and updates report with path.
        """
        report = self.generate_identity_report()
        chart_path = IdentityReportGenerator.generate_visual_chart(
            report=report,
            output_dir=output_dir,
            stable_threshold=self.drift_threshold,
            warning_threshold=self.warning_threshold,
        )
        report.html_report_path = str(chart_path)
        return chart_path

    def get_timeline(self) -> list[TimelinePoint]:
        """Returns the scrubbable timeline points."""
        return self.timeline.get_points()

    def get_drift_intervals(self, min_consecutive_frames: int = 1) -> list[dict[str, Any]]:
        """Returns contiguous video segments experiencing drift."""
        return self.timeline.get_drift_intervals(min_consecutive_frames=min_consecutive_frames)
