import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
from backend.app.identity.validator import IdentityValidator

@pytest.fixture
def validator():
    return IdentityValidator(job_id="test_job_123", drift_threshold=0.8)

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
    assert validator.detect_identity_drift(0.79) == True
    assert validator.detect_identity_drift(0.81) == False

def test_add_record_and_report(validator):
    # Add a good record
    validator.add_record(0, 0.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    # Add a drift record
    validator.add_record(1, 1.0, np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    report = validator.generate_identity_report()

    assert report.job_id == "test_job_123"
    assert report.total_frames_analyzed == 2
    assert report.drift_detected == True
    assert report.drift_occurrences == 1
    assert np.isclose(report.average_similarity, 0.5)
    assert np.isclose(report.min_similarity, 0.0)

def test_save_report(validator):
    validator.add_record(0, 0.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        report_path = validator.save_report(output_dir)
        
        assert report_path.exists()
        assert report_path.name == "identity_report_test_job_123.json"
        
        # Verify JSON is parseable and valid
        import json
        with open(report_path, "r") as f:
            data = json.load(f)
            assert data["job_id"] == "test_job_123"
            assert data["drift_detected"] == False

def test_generate_visual_charts(validator):
    validator.add_record(0, 0.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    validator.add_record(1, 1.0, np.array([1.0, 0.0]), np.array([0.9, 0.435]))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        chart_path = validator.generate_visual_charts(output_dir)
        
        assert chart_path.exists()
        assert chart_path.name == "identity_chart_test_job_123.html"
