"""
test_artifact_detector.py — Unit and integration tests for Phase 12:
- Interactive restoration fidelity tuning (w in [0.0, 1.0])
- Boundary Artifact Risk Detector (seam gradient discontinuity & CIE-Lab color divergence)
"""

import io
import json
import uuid
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.confidence.artifact_detector import (
    BoundaryArtifactDetector,
    BoundaryArtifactReport,
)
from main import UPLOADS_DIR, app

client = TestClient(app)


def test_artifact_detector_empty_frame():
    """Verify detector handles empty or None input gracefully without exceptions."""
    report = BoundaryArtifactDetector.detect_artifacts(None)
    assert isinstance(report, BoundaryArtifactReport)
    assert report.artifact_risk_level == "low"
    assert report.seam_gradient_score == 1.0
    assert report.color_discontinuity_score == 0.0


def test_artifact_detector_smooth_synthetic_frame():
    """Verify detector identifies smooth, seamless blending as low risk."""
    # Create smooth uniform image
    frame = np.full((300, 300, 3), 140, dtype=np.uint8)
    # Add gentle circle
    cv2.circle(frame, (150, 150), 60, (142, 142, 142), -1)
    # Apply Gaussian blur for seamless boundary
    frame = cv2.GaussianBlur(frame, (9, 9), 2.0)

    report = BoundaryArtifactDetector.detect_artifacts(
        composite_frame=frame,
        bbox=[90, 90, 210, 210],
    )

    assert isinstance(report, BoundaryArtifactReport)
    assert report.artifact_risk_level in ("low", "medium")
    assert report.color_discontinuity_score < 12.0
    assert report.coherence_score > 60.0


def test_artifact_detector_harsh_seam_and_color_mismatch():
    """Verify detector flags sharp boundary step jumps and extreme color divergence as high risk."""
    # Outer background is dark blue
    frame = np.full((300, 300, 3), (180, 50, 20), dtype=np.uint8)
    # Inner face region is bright green with harsh non-blended edge
    cv2.circle(frame, (150, 150), 60, (20, 240, 40), -1)

    report = BoundaryArtifactDetector.detect_artifacts(
        composite_frame=frame,
        bbox=[90, 90, 210, 210],
        band_thickness=5,
    )

    assert isinstance(report, BoundaryArtifactReport)
    assert report.artifact_risk_level == "high"
    assert report.color_discontinuity_score >= 15.0
    assert len(report.detected_artifacts) >= 1
    assert "Feathered" in report.recommendation or "restoration" in report.recommendation


def test_artifact_detector_endpoint():
    """Verify POST /confidence/detect-boundary-artifacts endpoint processes uploaded image."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, (180, 180, 180), -1)
    _, encoded = cv2.imencode(".jpg", img)

    response = client.post(
        "/confidence/detect-boundary-artifacts?x1=60&y1=60&x2=140&y2=140",
        files={"image": ("test_frame.jpg", encoded.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "seam_gradient_score" in data
    assert "color_discontinuity_score" in data
    assert "artifact_risk_level" in data
    assert "coherence_score" in data
    assert "recommendation" in data


def test_restoration_weight_query_validation():
    """Verify preview and process endpoints enforce w in [0.0, 1.0]."""
    session_id = uuid.uuid4().hex
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        (session_dir / "source_face.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01" + b"\x00" * 30)
        (session_dir / "target_video.mp4").write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 30)

        # Invalid weight > 1.0 should fail validation
        res_invalid = client.post(f"/preview?session_id={session_id}&restoration=gfpgan&restoration_weight=1.5")
        assert res_invalid.status_code == 422

        # Invalid weight < 0.0 should fail validation
        res_invalid_neg = client.post(f"/preview?session_id={session_id}&restoration=gfpgan&restoration_weight=-0.2")
        assert res_invalid_neg.status_code == 422

        # Valid weight 0.85 should succeed and queue job
        with patch("main.BackgroundTasks.add_task") as mock_add_task:
            res_valid = client.post(f"/preview?session_id={session_id}&restoration=gfpgan&restoration_weight=0.85")
            assert res_valid.status_code == 200
            assert "job_id" in res_valid.json()
            assert mock_add_task.called
    finally:
        import shutil

        shutil.rmtree(session_dir, ignore_errors=True)
