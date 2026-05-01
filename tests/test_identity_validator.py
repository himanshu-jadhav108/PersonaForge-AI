import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.identity.drift_detector import IdentityDriftDetector
from backend.app.identity.extractor import FaceEmbeddingExtractor
from backend.app.identity.models import DriftZone
from backend.app.identity.scoring import IdentityScorer
from backend.app.identity.timeline import IdentityTimeline
from backend.app.identity.validator import IdentityValidator
from main import OUTPUTS_DIR, app

client = TestClient(app)


@pytest.fixture
def validator():
    return IdentityValidator(job_id="test_job_123", drift_threshold=0.8)


# ─── Backward Compatibility & Core Validator Tests ─────────────────────────────


def test_compute_similarity_identical(validator):
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])
    score = validator.compute_similarity(emb1, emb2)
    assert np.isclose(score, 1.0)


def test_compute_similarity_orthogonal(validator):
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    score = validator.compute_similarity(emb1, emb2)
    assert np.isclose(score, 0.0)


def test_compute_similarity_opposite(validator):
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([-1.0, 0.0, 0.0])
    score = validator.compute_similarity(emb1, emb2)
    assert np.isclose(score, -1.0)


def test_detect_drift(validator):
    assert validator.detect_identity_drift(0.79) is True
    assert validator.detect_identity_drift(0.81) is False


def test_add_record_and_report(validator):
    # Add a good record
    validator.add_record(0, 0.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    # Add a drift record
    validator.add_record(1, 1.0, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    report = validator.generate_identity_report()

    assert report.job_id == "test_job_123"
    assert report.total_frames_analyzed == 2
    assert report.drift_detected is True
    assert report.drift_occurrences == 1
    assert np.isclose(report.average_similarity, 0.5)
    assert np.isclose(report.min_similarity, 0.0)
    assert len(report.timeline) == 2


def test_empty_validator_report(validator):
    report = validator.generate_identity_report()
    assert report.total_frames_analyzed == 0
    assert report.identity_score == 0.0
    assert report.drift_detected is False
    assert len(report.records) == 0
    assert len(report.timeline) == 0


def test_save_report(validator):
    validator.add_record(0, 0.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        report_path = validator.save_report(output_dir)

        assert report_path.exists()
        assert report_path.name == "identity_report_test_job_123.json"

        # Verify JSON is parseable and valid
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["job_id"] == "test_job_123"
            assert data["drift_detected"] is False


def test_generate_visual_charts(validator):
    validator.add_record(0, 0.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    validator.add_record(1, 1.0, np.array([1.0, 0.0]), np.array([0.9, 0.435]))

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        chart_path = validator.generate_visual_charts(output_dir)

        assert chart_path.exists()
        assert chart_path.name == "identity_chart_test_job_123.html"
        html_content = chart_path.read_text(encoding="utf-8")
        assert "plotly" in html_content.lower()


# ─── FaceEmbeddingExtractor Tests ──────────────────────────────────────────────


def test_extractor_normalize():
    vec = np.array([3.0, 4.0])
    norm_vec = FaceEmbeddingExtractor.normalize(vec)
    assert norm_vec is not None
    assert np.isclose(np.linalg.norm(norm_vec), 1.0)
    assert np.allclose(norm_vec, [0.6, 0.8])

    # Zero vector
    assert FaceEmbeddingExtractor.normalize(np.array([0.0, 0.0])) is None
    # None input
    assert FaceEmbeddingExtractor.normalize(None) is None


def test_extractor_extract_from_face():
    mock_face = MagicMock()
    mock_face.embedding = np.array([0.0, 5.0, 0.0])
    emb = FaceEmbeddingExtractor.extract_from_face(mock_face)
    assert emb is not None
    assert np.allclose(emb, [0.0, 1.0, 0.0])

    # Face without embedding
    mock_empty = MagicMock(spec=[])
    assert FaceEmbeddingExtractor.extract_from_face(mock_empty) is None


def test_extractor_extract_from_crop_edge_cases():
    assert FaceEmbeddingExtractor.extract_from_crop(None, None) is None
    assert FaceEmbeddingExtractor.extract_from_crop(np.empty((0, 0)), MagicMock()) is None

    # Mock app returning no faces
    mock_app = MagicMock()
    mock_app.get.return_value = []
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    assert FaceEmbeddingExtractor.extract_from_crop(crop, mock_app) is None


def test_extractor_compute_similarity_none():
    assert FaceEmbeddingExtractor.compute_similarity(None, np.array([1.0, 0.0])) == 0.0
    assert FaceEmbeddingExtractor.compute_similarity(np.array([1.0, 0.0]), None) == 0.0


# ─── IdentityDriftDetector Tests ───────────────────────────────────────────────


def test_drift_detector_zones():
    detector = IdentityDriftDetector(
        stable_threshold=0.80,
        warning_threshold=0.68,
        critical_threshold=0.55,
        delta_drop_threshold=0.20,
    )

    # Stable
    status_stable = detector.evaluate(current_similarity=0.85, rolling_similarity=0.86)
    assert status_stable.zone == DriftZone.STABLE
    assert status_stable.is_drift is False
    assert status_stable.is_sudden_drop is False

    # Warning
    status_warn = detector.evaluate(current_similarity=0.74, rolling_similarity=0.75)
    assert status_warn.zone == DriftZone.WARNING
    assert status_warn.is_drift is True
    assert status_warn.is_sudden_drop is False

    # Critical
    status_crit = detector.evaluate(current_similarity=0.50, rolling_similarity=0.55)
    assert status_crit.zone == DriftZone.CRITICAL
    assert status_crit.is_drift is True
    assert status_crit.is_sudden_drop is False


def test_drift_detector_sudden_drop():
    detector = IdentityDriftDetector(
        stable_threshold=0.80,
        warning_threshold=0.68,
        critical_threshold=0.55,
        delta_drop_threshold=0.20,
    )

    # Current similarity drops sharply from 0.88 rolling average to 0.65 (delta = 0.23 >= 0.20)
    status = detector.evaluate(current_similarity=0.65, rolling_similarity=0.88)
    assert status.is_sudden_drop is True
    assert status.is_drift is True
    assert np.isclose(status.drop_delta, 0.23)


# ─── IdentityScorer Tests ──────────────────────────────────────────────────────


def test_scorer_rolling_averages():
    values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    rolling = IdentityScorer.compute_rolling_averages(values, window_size=3)
    assert len(rolling) == 6
    assert np.isclose(rolling[0], 1.0)
    assert np.isclose(rolling[1], 0.9)  # (1.0 + 0.8) / 2
    assert np.isclose(rolling[2], 0.8)  # (1.0 + 0.8 + 0.6) / 3
    assert np.isclose(rolling[3], 0.6)  # (0.8 + 0.6 + 0.4) / 3


def test_scorer_compute_statistics():
    sims = [0.70, 0.80, 0.90]
    stats = IdentityScorer.compute_statistics(sims)
    assert np.isclose(stats["mean"], 0.80)
    assert np.isclose(stats["min"], 0.70)
    assert np.isclose(stats["max"], 0.90)
    assert stats["std"] > 0.0

    empty_stats = IdentityScorer.compute_statistics([])
    assert empty_stats["mean"] == 0.0


def test_scorer_calculate_identity_score_calibrated():
    # Perfect high match: avg_sim >= 0.70
    score_high = IdentityScorer.calculate_identity_score([0.80, 0.85, 0.90])
    assert score_high == 100.0

    # Low match at baseline (0.20)
    score_low = IdentityScorer.calculate_identity_score([0.20, 0.20])
    assert score_low == 0.0

    # Penalized score for sudden drops and critical frames
    score_penalized = IdentityScorer.calculate_identity_score(
        [0.75, 0.75],
        critical_count=2,
        sudden_drops=1,
    )
    # raw_norm = 100.0, penalty = min(20.0, 1*2.5 + 2*1.0) = 4.5
    assert score_penalized == 95.5


# ─── IdentityTimeline Tests ────────────────────────────────────────────────────


def test_identity_timeline_intervals():
    timeline = IdentityTimeline()
    # 3 stable frames
    for i in range(3):
        timeline.add_point(i, i * 0.1, 0.85, 0.85, DriftZone.STABLE)
    # 2 warning frames
    timeline.add_point(3, 0.3, 0.72, 0.80, DriftZone.WARNING)
    timeline.add_point(4, 0.4, 0.70, 0.75, DriftZone.WARNING)
    # 1 critical frame
    timeline.add_point(5, 0.5, 0.50, 0.65, DriftZone.CRITICAL)
    # 2 stable frames
    timeline.add_point(6, 0.6, 0.85, 0.75, DriftZone.STABLE)
    timeline.add_point(7, 0.7, 0.88, 0.80, DriftZone.STABLE)

    intervals = timeline.get_drift_intervals(min_consecutive_frames=2)
    assert len(intervals) == 1
    drift_seg = intervals[0]
    assert drift_seg["start_frame"] == 3
    assert drift_seg["end_frame"] == 5
    assert drift_seg["frame_count"] == 3
    assert drift_seg["severity"] == DriftZone.CRITICAL.value
    assert np.isclose(drift_seg["min_similarity"], 0.50)


# ─── Identity FastAPI Endpoints Tests ──────────────────────────────────────────


def test_get_identity_report_endpoint():
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_id = "test_endpoint_job"
    report_file = reports_dir / f"identity_report_{test_id}.json"

    try:
        report_file.write_text(json.dumps({"job_id": test_id, "identity_score": 92.5}), encoding="utf-8")

        response = client.get(f"/identity/report/{test_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == test_id
        assert data["identity_score"] == 92.5

        # Non-existent job
        resp_404 = client.get("/identity/report/non_existent_12345")
        assert resp_404.status_code == 404
    finally:
        report_file.unlink(missing_ok=True)


def test_get_identity_chart_endpoint():
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_id = "test_chart_job"
    chart_file = reports_dir / f"identity_chart_{test_id}.html"

    try:
        chart_file.write_text("<html><body>Identity Chart</body></html>", encoding="utf-8")

        response = client.get(f"/identity/chart/{test_id}")
        assert response.status_code == 200
        assert "Identity Chart" in response.text
        assert "text/html" in response.headers["content-type"]

        # Non-existent chart
        resp_404 = client.get("/identity/chart/non_existent_12345")
        assert resp_404.status_code == 404
    finally:
        chart_file.unlink(missing_ok=True)
