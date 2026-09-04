"""
tests/test_frontend_endpoints.py

Validates frontend support endpoints for Phase 7:
- GET / (frontend delivery)
- GET /restoration/status (Safeguard 2 transparency)
- POST /cancel/{job_id} (graceful job cancellation)
"""

import uuid

from fastapi.testclient import TestClient

from main import app, db

client = TestClient(app)


def test_serve_frontend():
    res = client.get("/")
    assert res.status_code == 200
    assert "PersonaForge AI" in res.text
    assert "html" in res.headers.get("content-type", "")


def test_restoration_status_endpoint():
    res = client.get("/restoration/status?adapter=gfpgan")
    assert res.status_code == 200
    data = res.json()
    assert "ai_restoration" in data
    assert "classic_enhancement" in data
    assert "status_message" in data

    # Test classic adapter
    res_classic = client.get("/restoration/status?adapter=classic")
    assert res_classic.status_code == 200
    data_classic = res_classic.json()
    assert data_classic["is_ai"] is False
    assert "Classic" in data_classic["active_restorer"]


def test_cancel_job_lifecycle():
    job_id = uuid.uuid4().hex
    db.insert_job({
        "id": job_id,
        "session_id": "test_session_id",
        "kind": "full",
        "status": "running",
        "stage": "processing",
        "progress": 45,
        "message": "Processing frames...",
        "created_at": "2026-09-04T12:00:00Z"
    })

    # Cancel active job
    res = client.post(f"/cancel/{job_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "cancelled"

    # Check DB reflects cancelled state
    job = db.get_job(job_id)
    assert job["status"] == "cancelled"
    assert "cancelled" in job["message"].lower()

    # Idempotent cancel attempt
    res_again = client.post(f"/cancel/{job_id}")
    assert res_again.status_code == 200
    data_again = res_again.json()
    assert data_again["status"] == "cancelled"


def test_cancel_nonexistent_job():
    res = client.post("/cancel/non_existent_job_12345")
    assert res.status_code == 404
