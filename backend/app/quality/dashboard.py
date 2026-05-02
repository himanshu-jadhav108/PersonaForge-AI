import logging
from pathlib import Path

import plotly.graph_objects as go

from backend.app.quality.models import QualityReport

logger = logging.getLogger("personaforge.quality")


def generate_dashboard(report: QualityReport, output_dir: Path, image_name: str) -> Path:
    """
    Generates an interactive Plotly radar chart dashboard for face quality metrics.
    Saves it as an HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = output_dir / f"quality_dashboard_{image_name}.html"

    categories = ["Blur", "Lighting", "Sharpness", "Face Angle", "Occlusion", "Face Size"]
    metrics = report.metrics

    values = [
        metrics.blur,
        metrics.lighting if metrics.lighting is not None else 50.0,
        metrics.sharpness,
        metrics.face_angle,
        metrics.occlusion,
        metrics.face_size,
    ]

    # Close the radar polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="Quality Metrics",
        line_color="#6366f1",
        fillcolor="rgba(99, 102, 241, 0.2)",
    ))

    # Add recommendations as annotations
    annotations = []
    y_pos = -0.15
    for rec in report.recommendations:
        annotations.append({
            "x": 0.5,
            "y": y_pos,
            "xref": "paper",
            "yref": "paper",
            "text": f"💡 {rec}",
            "showarrow": False,
            "font": {"size": 12, "color": "#4b5563"},
        })
        y_pos -= 0.05

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
            }
        },
        showlegend=False,
        title=f"Face Quality Score: {report.quality_score}/100",
        template="plotly_white",
        annotations=annotations,
    )

    fig.write_html(str(chart_path))
    logger.info("Quality dashboard saved to %s", chart_path)

    return chart_path
