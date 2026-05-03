import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.confidence.boundary import BoundaryCoherenceEvaluator
from backend.app.confidence.models import IntegrityTier
from backend.app.confidence.scorer import PersonaForgeIntegrityScorer
from main import OUTPUTS_DIR, app

client = TestClient(app)


# ─── BoundaryCoherenceEvaluator Tests ──────────────────────────────────────────


def test_boundary_coherence_evaluator_smooth_vs_sharp():
    # Synthetic image: 200x200
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)

    # Completely uniform image -> zero gradient everywhere
    ratio, score = BoundaryCoherenceEvaluator.evaluate_boundary_coherence(img, mask)
    assert ratio <= 1.0
    assert score == 100.0

    # Image with sharp step at mask boundary
    sharp_img = img.copy()
    cv2.circle(sharp_img, (100, 100), 50, 255, -1)  # white face on gray background
    sharp_ratio, sharp_score = BoundaryCoherenceEvaluator.evaluate_boundary_coherence(sharp_img, mask)
    assert sharp_ratio > 1.0
    assert sharp_score < 100.0


def test_boundary_coherence_evaluator_edge_cases():
    ratio, score = BoundaryCoherenceEvaluator.evaluate_boundary_coherence(None, None)
    assert ratio == 1.0
    assert score == 100.0

    ratio, score = BoundaryCoherenceEvaluator.evaluate_boundary_coherence(np.empty((0, 0)), np.empty((0, 0)))
    assert ratio == 1.0
    assert score == 100.0


# ─── Component Normalization Tests ─────────────────────────────────────────────


def test_compute_norm_sim_calibration():
    # Random face boundary (0.20) -> 0%
    assert PersonaForgeIntegrityScorer.compute_norm_sim(0.20) == 0.0
    assert PersonaForgeIntegrityScorer.compute_norm_sim(0.10) == 0.0

    # Clean match threshold (0.70) -> 100%
    assert PersonaForgeIntegrityScorer.compute_norm_sim(0.70) == 100.0
    assert PersonaForgeIntegrityScorer.compute_norm_sim(0.85) == 100.0

    # Midpoint (0.45) -> 50%
    assert np.isclose(PersonaForgeIntegrityScorer.compute_norm_sim(0.45), 50.0)


def test_compute_norm_sharp_calibration():
    # Below baseline (25) -> 0%
    assert PersonaForgeIntegrityScorer.compute_norm_sharp(25.0) == 0.0
    assert PersonaForgeIntegrityScorer.compute_norm_sharp(10.0) == 0.0

    # High definition target (300) -> 100%
    assert PersonaForgeIntegrityScorer.compute_norm_sharp(300.0) == 100.0
    assert PersonaForgeIntegrityScorer.compute_norm_sharp(600.0) == 100.0

    # Zero variance -> 0%
    assert PersonaForgeIntegrityScorer.compute_norm_sharp(0.0) == 0.0


def test_compute_norm_stability_calibration():
    # Zero jitter -> 100%
    assert PersonaForgeIntegrityScorer.compute_norm_stability(0.0) == 100.0

    # Ceiling jitter (0.20 IOD) -> 0%
    assert PersonaForgeIntegrityScorer.compute_norm_stability(0.20) == 0.0
    assert PersonaForgeIntegrityScorer.compute_norm_stability(0.25) == 0.0

    # Typical head motion (0.05 IOD) -> 75%
    assert np.isclose(PersonaForgeIntegrityScorer.compute_norm_stability(0.05), 75.0)


def test_compute_norm_boundary_calibration():
    # Seam ratio <= 1.0 -> 100%
    assert PersonaForgeIntegrityScorer.compute_norm_boundary(0.9) == 100.0
    assert PersonaForgeIntegrityScorer.compute_norm_boundary(1.0) == 100.0

    # Hard seam ratio >= 2.5 -> 0%
    assert PersonaForgeIntegrityScorer.compute_norm_boundary(2.5) == 0.0
    assert PersonaForgeIntegrityScorer.compute_norm_boundary(3.0) == 0.0


def test_compute_norm_detection_calibration():
    assert PersonaForgeIntegrityScorer.compute_norm_detection(0.95) == 95.0
    assert PersonaForgeIntegrityScorer.compute_norm_detection(95.0) == 95.0


# ─── Composite Score & Report Tests ────────────────────────────────────────────


def test_evaluate_three_components_baseline():
    # Standard formula from docs/metrics.md Section 5:
    # 0.40 * NormSim + 0.35 * NormSharp + 0.25 * NormStability
    report = PersonaForgeIntegrityScorer.evaluate(
        job_id="test_job_3comp",
        cosine_similarity=0.70,     # NormSim = 100.0 (contrib = 40.0)
        laplacian_variance=300.0,   # NormSharp = 100.0 (contrib = 35.0)
        jitter_iod=0.0,             # NormStability = 100.0 (contrib = 25.0)
    )

    assert report.integrity_score == 100.0
    assert report.tier == IntegrityTier.EXCELLENT
    assert report.breakdown.identity.weighted_contribution == 40.0
    assert report.breakdown.sharpness.weighted_contribution == 35.0
    assert report.breakdown.temporal_stability.weighted_contribution == 25.0
    assert report.breakdown.boundary_coherence is None
    assert report.breakdown.detection_confidence is None
    assert "IDENTITY_FIDELITY_HIGH" in report.badges
    assert "STUDIO_SHARPNESS" in report.badges
    assert "TEMPORALLY_STABLE" in report.badges


def test_evaluate_five_components_full():
    report = PersonaForgeIntegrityScorer.evaluate(
        job_id="test_job_5comp",
        cosine_similarity=0.60,
        laplacian_variance=180.0,
        jitter_iod=0.03,
        boundary_ratio=1.10,
        det_score=0.99,
    )

    assert 0.0 <= report.integrity_score <= 100.0
    assert report.breakdown.boundary_coherence is not None
    assert report.breakdown.detection_confidence is not None
    assert len(report.explanations) == 5
    assert len(report.recommendations) >= 1


def test_evaluate_degraded_tier_and_warning_badges():
    report = PersonaForgeIntegrityScorer.evaluate(
        job_id="test_job_degraded",
        cosine_similarity=0.25,     # Very low similarity
        laplacian_variance=15.0,    # High blur
        jitter_iod=0.19,            # High jitter
        boundary_ratio=2.6,         # Seam artifact
        det_score=0.40,             # Low detection confidence
    )

    assert report.tier == IntegrityTier.DEGRADED
    assert "DRIFT_RISK" in report.badges
    assert "MOTION_BLUR" in report.badges
    assert "LANDMARK_JITTER" in report.badges
    assert "SEAM_ARTIFACT" in report.badges
    assert "OCCLUSION_WARNING" in report.badges


# ─── API Endpoints Tests ───────────────────────────────────────────────────────


def test_evaluate_integrity_endpoint():
    response = client.post(
        "/integrity/evaluate",
        params={
            "job_id": "test_api_job",
            "cosine_similarity": 0.65,
            "laplacian_variance": 200.0,
            "jitter_iod": 0.025,
            "boundary_ratio": 1.15,
            "det_score": 0.98,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test_api_job"
    assert data["integrity_score"] > 70.0
    assert "tier" in data
    assert "breakdown" in data


def test_get_integrity_report_endpoint():
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_id = "test_integ_get_job"
    report_file = reports_dir / f"integrity_report_{test_id}.json"

    try:
        report_file.write_text(
            json.dumps({"job_id": test_id, "integrity_score": 88.5, "tier": "EXCELLENT"}),
            encoding="utf-8",
        )

        response = client.get(f"/integrity/report/{test_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == test_id
        assert data["integrity_score"] == 88.5

        # 404 for unknown job
        resp_404 = client.get("/integrity/report/non_existent_99999")
        assert resp_404.status_code == 404
    finally:
        report_file.unlink(missing_ok=True)
