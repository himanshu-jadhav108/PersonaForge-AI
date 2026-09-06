"""
tests/test_system_security.py — Unit and integration tests for Phase 9 (Security, Retention, and Job Management).
"""

import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.security import (
    RetentionManager,
    sanitize_filename,
    validate_file_security,
    validate_media_magic_bytes,
    validate_session_id,
)
from main import app
from utils.database import JobDB

client = TestClient(app)


def test_session_id_validation():
    """Verify strict session ID validation and path-traversal prevention."""
    valid_hex32 = uuid.uuid4().hex
    valid_uuid4 = str(uuid.uuid4())

    assert validate_session_id(valid_hex32) is True
    assert validate_session_id(valid_uuid4) is True

    # Path traversal and injection attacks
    assert validate_session_id("../../etc/passwd") is False
    assert validate_session_id("..\\..\\windows\\system32") is False
    assert validate_session_id(f"{valid_hex32}/subpath") is False
    assert validate_session_id(f"{valid_hex32}\x00traversal") is False
    assert validate_session_id("") is False
    assert validate_session_id(None) is False
    assert validate_session_id("short_string") is False


def test_filename_sanitization():
    """Verify filename sanitization removes path components and dangerous characters."""
    assert sanitize_filename("../../../malicious.mp4") == "malicious.mp4"
    assert sanitize_filename("..\\..\\windows\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("my test video (1) [hd]!.mp4") == "my_test_video_1_hd_.mp4"
    assert sanitize_filename("", default="fallback.png") == "fallback.png"
    assert sanitize_filename("...", default="fallback.mp4") == "fallback.mp4"


def test_media_magic_bytes_and_security(tmp_path: Path):
    """Verify binary signature inspection for images and videos."""
    # Valid PNG
    png_file = tmp_path / "valid.png"
    png_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 30)
    assert validate_media_magic_bytes(png_file, "image") is True

    valid, _ = validate_file_security(png_file, "image", max_bytes=1024 * 1024)
    assert valid is True

    # Valid JPEG
    jpg_file = tmp_path / "valid.jpg"
    jpg_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01" + b"\x00" * 30)
    assert validate_media_magic_bytes(jpg_file, "image") is True

    # Valid MP4
    mp4_file = tmp_path / "valid.mp4"
    mp4_file.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 30)
    assert validate_media_magic_bytes(mp4_file, "video") is True

    # Spoofed file (text pretending to be JPG)
    fake_jpg = tmp_path / "fake.jpg"
    fake_jpg.write_text("This is plain text with a jpg extension", encoding="utf-8")
    assert validate_media_magic_bytes(fake_jpg, "image") is False

    valid_fake, reason_fake = validate_file_security(fake_jpg, "image")
    assert valid_fake is False
    assert "Invalid image signature" in reason_fake

    # Empty file
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    valid_empty, reason_empty = validate_file_security(empty_file, "video")
    assert valid_empty is False
    assert "empty" in reason_empty.lower()


def test_retention_manager_cleanup(tmp_path: Path):
    """Verify storage cleanup purges expired artifacts and leaves recent artifacts."""
    base_dir = tmp_path / "personaforge_test"
    base_dir.mkdir()
    uploads = base_dir / "uploads"
    outputs = base_dir / "outputs"
    frames = base_dir / "temp_frames"
    frames.mkdir()
    db_file = base_dir / "test_jobs.db"
    db = JobDB(str(db_file))

    mgr = RetentionManager(base_dir=base_dir, retention_hours=1.0, db=db)

    # Create old upload (> 2 hours old)
    old_upload = uploads / "old_session"
    old_upload.mkdir(parents=True)
    (old_upload / "test.txt").write_text("old data", encoding="utf-8")
    old_mtime = time.time() - 7500
    import os

    os.utime(str(old_upload), (old_mtime, old_mtime))
    os.utime(str(old_upload / "test.txt"), (old_mtime, old_mtime))

    # Create fresh upload (< 5 mins old)
    fresh_upload = uploads / "fresh_session"
    fresh_upload.mkdir(parents=True)
    (fresh_upload / "test.txt").write_text("fresh data", encoding="utf-8")

    # Create old output
    old_out = outputs / "old_out.mp4"
    old_out.write_bytes(b"old video content")
    os.utime(str(old_out), (old_mtime, old_mtime))

    # Create old DB job
    db.insert_job(
        {
            "id": "job_old",
            "session_id": "sess_old",
            "kind": "full",
            "status": "done",
            "stage": "completed",
            "progress": 100,
            "message": "done",
            "output": "old_out.mp4",
            "file_size_mb": 1.0,
            "device": "CPU",
            "mode": "cpu",
            "similarity_score": 0.8,
            "orientation": "landscape",
            "input_width": 1280,
            "input_height": 720,
            "resize_mode": "maintain",
            "created_at": "2020-01-01T00:00:00Z",
        }
    )

    # Create fresh DB job
    db.insert_job(
        {
            "id": "job_fresh",
            "session_id": "sess_fresh",
            "kind": "full",
            "status": "running",
            "stage": "processing",
            "progress": 50,
            "message": "working",
            "output": None,
            "file_size_mb": None,
            "device": "CPU",
            "mode": "cpu",
            "similarity_score": 0.8,
            "orientation": "landscape",
            "input_width": 1280,
            "input_height": 720,
            "resize_mode": "maintain",
            "created_at": "2026-09-06T12:00:00Z",
        }
    )

    # Pre-cleanup telemetry check
    stats = mgr.get_storage_stats()
    assert stats.retention_hours == 1.0
    assert stats.uploads.file_count >= 2

    # Execute cleanup
    report = mgr.perform_cleanup(retention_hours=1.0)
    assert report.deleted_uploads >= 1
    assert report.deleted_outputs >= 1
    assert report.deleted_jobs >= 1
    assert report.freed_bytes > 0

    # Verify old items deleted, fresh items kept
    assert not old_upload.exists()
    assert not old_out.exists()
    assert fresh_upload.exists()
    assert db.get_job("job_old") is None
    assert db.get_job("job_fresh") is not None


def test_system_endpoints():
    """Test /system/retention, /system/cleanup, and /system/health REST endpoints."""
    # GET /system/retention
    res_ret = client.get("/system/retention")
    assert res_ret.status_code == 200
    ret_data = res_ret.json()
    assert "retention_hours" in ret_data
    assert "total_storage_mb" in ret_data
    assert "jobs_by_status" in ret_data

    # POST /system/cleanup
    res_clean = client.post("/system/cleanup?retention_hours=48&force_all_temp=false")
    assert res_clean.status_code == 200
    clean_data = res_clean.json()
    assert clean_data["retention_hours_used"] == 48.0
    assert "freed_mb" in clean_data

    # GET /system/health
    res_health = client.get("/system/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert "status" in health_data
    assert "models_verified" in health_data


def test_job_cancellation_workflow():
    """Verify job cancellation lifecycle, status codes, and idempotence."""
    # 404 for nonexistent job
    res_404 = client.post("/cancel/nonexistent_job_12345")
    assert res_404.status_code == 404

    # Create a job in DB
    from main import db

    test_jid = uuid.uuid4().hex
    db.insert_job(
        {
            "id": test_jid,
            "session_id": "test_session_cancel",
            "kind": "full",
            "status": "processing",
            "stage": "processing",
            "progress": 45,
            "message": "Swapping faces...",
            "output": None,
            "file_size_mb": None,
            "device": "GPU",
            "mode": "gpu",
            "similarity_score": 0.82,
            "orientation": "portrait",
            "input_width": 720,
            "input_height": 1280,
            "resize_mode": "maintain",
            "created_at": "2026-09-06T12:00:00Z",
        }
    )

    # Cancel the active job
    res_cancel = client.post(f"/cancel/{test_jid}")
    assert res_cancel.status_code == 200
    cancel_data = res_cancel.json()
    assert cancel_data["status"] == "cancelled"

    # Confirm in DB
    job_in_db = db.get_job(test_jid)
    assert job_in_db["status"] == "cancelled"
    assert job_in_db["stage"] == "cancelled"
    assert "completed_at" in job_in_db and job_in_db["completed_at"] is not None

    # Idempotence: subsequent cancel returns current cancelled status
    res_again = client.post(f"/cancel/{test_jid}")
    assert res_again.status_code == 200
    assert "already cancelled" in res_again.json()["message"]
