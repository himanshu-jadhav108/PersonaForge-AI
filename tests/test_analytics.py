import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_overview():
    response = client.get("/analytics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_jobs" in data
    assert "success_rate" in data
    assert "failed_jobs" in data
    assert "average_processing_time" in data

def test_get_performance():
    response = client.get("/analytics/performance")
    assert response.status_code == 200
    data = response.json()
    assert "job_ids" in data
    assert "processing_times" in data
    assert "queue_lengths" in data

def test_get_system():
    response = client.get("/analytics/system")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_utilization" in data
    assert "memory_utilization" in data
    assert "gpu_utilization" in data

def test_get_identity():
    response = client.get("/analytics/identity")
    assert response.status_code == 200
    data = response.json()
    assert "trends" in data
    assert isinstance(data["trends"], list)
