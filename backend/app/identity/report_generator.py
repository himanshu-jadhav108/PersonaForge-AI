import json
import logging
from pathlib import Path

import plotly.graph_objects as go

from backend.app.identity.models import IdentityReport

logger = logging.getLogger("personaforge.identity.report_generator")


class IdentityReportGenerator:
    """
    Generates standardized JSON reports and interactive Plotly visual dashboards
    for identity consistency over video timelines.
    """

    @classmethod
    def save_json_report(cls, report: IdentityReport, output_dir: Path) -> Path:
        """
        Saves the structured JSON identity report to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"identity_report_{report.job_id}.json"

        report_dict = report.model_dump()
        report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        logger.info("Saved identity JSON report to %s", report_path)
        return report_path

    @classmethod
    def generate_visual_chart(
        cls,
        report: IdentityReport,
        output_dir: Path,
        stable_threshold: float = 0.80,
        warning_threshold: float = 0.68,
    ) -> Path:
        """
        Generates an interactive Plotly HTML report with shaded drift zones,
        rolling average line, and marked sudden drop points.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / f"identity_chart_{report.job_id}.html"

        if not report.records and not report.timeline:
            logger.warning("No records available to plot identity chart for %s", report.job_id)
            return chart_path

        # Gather points
        if report.timeline:
            frames = [pt.frame_index for pt in report.timeline]
            sims = [pt.similarity for pt in report.timeline]
            rolling = [pt.rolling_similarity for pt in report.timeline]
            sudden_drop_x = [pt.frame_index for pt in report.timeline if pt.is_sudden_drop]
            sudden_drop_y = [pt.similarity for pt in report.timeline if pt.is_sudden_drop]
        else:
            frames = [r.frame_index for r in report.records]
            sims = [r.similarity_score for r in report.records]
            rolling = [r.rolling_similarity if r.rolling_similarity is not None else r.similarity_score for r in report.records]
            sudden_drop_x = [r.frame_index for r in report.records if getattr(r, 'sudden_drop', False)]
            sudden_drop_y = [r.similarity_score for r in report.records if getattr(r, 'sudden_drop', False)]

        fig = go.Figure()

        # 1. Background drift zones (Shaded horizontal bands)
        fig.add_hrect(
            y0=stable_threshold, y1=1.0,
            fillcolor="rgba(34, 197, 94, 0.12)", line_width=0,
            annotation_text="Stable Zone (≥0.80)", annotation_position="top left",
        )
        fig.add_hrect(
            y0=warning_threshold, y1=stable_threshold,
            fillcolor="rgba(234, 179, 8, 0.12)", line_width=0,
            annotation_text="Warning Zone (0.68–0.80)", annotation_position="left",
        )
        fig.add_hrect(
            y0=0.0, y1=warning_threshold,
            fillcolor="rgba(239, 68, 68, 0.12)", line_width=0,
            annotation_text="Critical Drift Zone (<0.68)", annotation_position="bottom left",
        )

        # 2. Raw Cosine Similarity Trace
        fig.add_trace(go.Scatter(
            x=frames,
            y=sims,
            mode="lines+markers",
            name="Frame Similarity",
            line={"color": "#3b82f6", "width": 2},
            marker={"size": 4},
            hovertemplate="Frame %{x}<br>Similarity: %{y:.3f}<extra></extra>",
        ))

        # 3. Rolling Moving Average Trace
        fig.add_trace(go.Scatter(
            x=frames,
            y=rolling,
            mode="lines",
            name="Rolling Trend (5-frame)",
            line={"color": "#8b5cf6", "width": 2, "dash": "dash"},
            hovertemplate="Frame %{x}<br>Rolling Avg: %{y:.3f}<extra></extra>",
        ))

        # 4. Sudden Drop Markers (if any)
        if sudden_drop_x:
            fig.add_trace(go.Scatter(
                x=sudden_drop_x,
                y=sudden_drop_y,
                mode="markers",
                name="Sudden Drop (Δ ≥ 0.20)",
                marker={"color": "#ef4444", "size": 10, "symbol": "diamond"},
                hovertemplate="Frame %{x}<br>Sudden Drop: %{y:.3f}<extra></extra>",
            ))

        # 5. Threshold Guide Lines
        fig.add_hline(
            y=stable_threshold,
            line_dash="dot",
            line_color="rgba(34, 197, 94, 0.7)",
            annotation_text=f"Stable ({stable_threshold})",
            annotation_position="bottom right",
        )
        fig.add_hline(
            y=warning_threshold,
            line_dash="dot",
            line_color="rgba(234, 179, 8, 0.7)",
            annotation_text=f"Warning ({warning_threshold})",
            annotation_position="bottom right",
        )

        # Layout styling
        status_label = "EXCELLENT" if report.identity_score >= 85 else ("STABLE" if not report.drift_detected else "DRIFT DETECTED")
        fig.update_layout(
            title=(
                f"PersonaForge Identity Consistency (Job: {report.job_id[:8]})<br>"
                f"<sup>Identity Score: {report.identity_score}% [{status_label}] | "
                f"Avg Sim: {report.average_similarity:.3f} | "
                f"Min Sim: {report.min_similarity:.3f} | "
                f"Drift Frames: {report.drift_occurrences}/{report.total_frames_analyzed}</sup>"
            ),
            xaxis_title="Video Frame Index",
            yaxis_title="ArcFace Cosine Similarity",
            yaxis_range=[0.0, 1.05],
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            hovermode="x unified",
        )

        fig.write_html(str(chart_path))
        logger.info("Saved interactive identity chart to %s", chart_path)
        return chart_path
