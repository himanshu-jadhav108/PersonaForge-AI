import plotly.graph_objects as go
from pathlib import Path
import logging

from backend.app.selection.models import SelectionReport

logger = logging.getLogger("personaforge.selection")

def generate_dashboard(report: SelectionReport, output_dir: Path, filename_prefix: str) -> Path:
    """
    Generates a Plotly dashboard comparing the clustered faces based on their metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chart_path = output_dir / f"selection_dashboard_{filename_prefix}.html"
    
    face_ids = [p.face_id[:8] for p in report.profiles]
    
    # Scale areas for better visualization
    max_area = max([p.average_area for p in report.profiles]) if report.profiles else 1.0
    if max_area == 0: max_area = 1.0
    
    areas = [(p.average_area / max_area) * 100.0 for p in report.profiles]
    visibilities = [p.visibility_duration for p in report.profiles]
    speaking_scores = [min(100.0, p.speaking_score) for p in report.profiles] # clip for visual
    confidences = [p.detection_confidence for p in report.profiles]
    
    fig = go.Figure(data=[
        go.Bar(name='Relative Area', x=face_ids, y=areas),
        go.Bar(name='Visibility (%)', x=face_ids, y=visibilities),
        go.Bar(name='Speaking Score (Scaled)', x=face_ids, y=speaking_scores),
        go.Bar(name='Detection Confidence', x=face_ids, y=confidences)
    ])
    
    # Change the bar mode
    fig.update_layout(barmode='group')
    
    selected_short_id = report.selected_face_id[:8]
    
    fig.update_layout(
        title=f"Face Selection Candidates (Mode: {report.selection_mode.value})<br>Optimal Pick: {selected_short_id} (Confidence: {report.confidence_score}%)",
        xaxis_title="Face Identity",
        yaxis_title="Score / Percentage",
        template="plotly_white"
    )
    
    fig.write_html(str(chart_path))
    logger.info(f"Selection dashboard saved to {chart_path}")
    
    return chart_path
