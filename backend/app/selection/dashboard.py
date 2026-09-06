import logging
from pathlib import Path

import plotly.graph_objects as go

from backend.app.selection.models import SelectionReport

logger = logging.getLogger("personaforge.selection")


def generate_dashboard(report: SelectionReport, output_dir: Path, filename_prefix: str) -> Path:
    """
    Generates a Plotly dashboard comparing the clustered faces based on their metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"selection_dashboard_{filename_prefix}.html"

    labels = [f"{getattr(p, 'person_label', 'Person')} ({p.face_id[:8]})" for p in report.profiles]

    # Scale areas for better visualization
    max_area = max([p.average_area for p in report.profiles]) if report.profiles else 1.0
    if max_area == 0:
        max_area = 1.0

    areas = [(p.average_area / max_area) * 100.0 for p in report.profiles]
    visibilities = [p.visibility_duration for p in report.profiles]
    speaking_scores = [min(100.0, p.speaking_score) for p in report.profiles]
    confidences = [p.detection_confidence for p in report.profiles]

    fig = go.Figure(
        data=[
            go.Bar(name="Relative Area", x=labels, y=areas),
            go.Bar(name="Visibility (%)", x=labels, y=visibilities),
            go.Bar(name="Speaking Score (Scaled)", x=labels, y=speaking_scores),
            go.Bar(name="Detection Confidence", x=labels, y=confidences),
        ]
    )

    fig.update_layout(barmode="group")

    optimal_name = report.selected_person_label or report.selected_face_id[:8]

    fig.update_layout(
        title=(
            f"Face Selection Candidates (Mode: {report.selection_mode.value})<br>"
            f"Optimal Pick: {optimal_name} (Confidence: {report.confidence_score}%)"
        ),
        xaxis_title="Face Identity",
        yaxis_title="Score / Percentage",
        template="plotly_white",
    )

    fig.write_html(str(chart_path))
    logger.info("Selection dashboard saved to %s", chart_path)

    return chart_path
