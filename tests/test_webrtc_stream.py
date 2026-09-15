"""
test_webrtc_stream.py — Unit and integration tests for Phase 13:
- WebRTC streaming endpoints (/realtime/offer, /realtime/connections, /realtime/stop, /realtime/stats)
- Lifespan modernization and clean peer connection lifecycle shutdown
"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.realtime.router import AIORTC_AVAILABLE, close_peer_connections
from main import UPLOADS_DIR, app

client = TestClient(app)


def test_realtime_connections_status():
    """Verify /realtime/connections returns diagnostic telemetry."""
    response = client.get("/realtime/connections")
    assert response.status_code == 200
    data = response.json()
    assert "active_connections" in data
    assert "aiortc_available" in data
    assert "is_processor_initialized" in data
    assert isinstance(data["active_connections"], int)


def test_realtime_stats_endpoint():
    """Verify /realtime/stats returns processor telemetry or idle status."""
    response = client.get("/realtime/stats")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "fps" in data


def test_realtime_stop_endpoint():
    """Verify /realtime/stop executes cleanup without error."""
    response = client.post("/realtime/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "stopped"
    assert "closed_connections" in data


def test_realtime_offer_missing_source_image():
    """Verify /realtime/offer rejects non-existent source images with 400."""
    fake_sdp = "v=0\r\no=- 12345 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
    response = client.post(
        "/realtime/offer",
        json={
            "sdp": fake_sdp,
            "type": "offer",
            "source_image_path": "non_existent/path/to/face.jpg",
        },
    )
    if AIORTC_AVAILABLE:
        assert response.status_code == 400
        assert "Source image not found" in response.json()["detail"]
    else:
        assert response.status_code == 503


def test_realtime_offer_session_id_resolution():
    """Verify /realtime/offer resolves source face using session_id."""
    session_id = uuid.uuid4().hex
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    src_file = session_dir / "source_face.jpg"
    src_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01" + b"\x00" * 30)

    fake_sdp = "v=0\r\no=- 12345 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"

    try:
        if AIORTC_AVAILABLE:
            with (
                patch("backend.app.realtime.router.RTCPeerConnection") as mock_pc_cls,
                patch("backend.app.realtime.router.RealTimeProcessor"),
            ):
                mock_pc = MagicMock()
                mock_pc.setRemoteDescription = AsyncMock()
                mock_pc.createAnswer = AsyncMock(return_value=MagicMock())
                mock_pc.setLocalDescription = AsyncMock()
                mock_pc.localDescription = MagicMock(sdp="answer_sdp", type="answer")
                mock_pc_cls.return_value = mock_pc

                response = client.post(
                    "/realtime/offer",
                    json={
                        "sdp": fake_sdp,
                        "type": "offer",
                        "session_id": session_id,
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert data["sdp"] == "answer_sdp"
                assert data["type"] == "answer"
        else:
            response = client.post(
                "/realtime/offer",
                json={
                    "sdp": fake_sdp,
                    "type": "offer",
                    "session_id": session_id,
                },
            )
            assert response.status_code == 503
    finally:
        import shutil

        shutil.rmtree(session_dir, ignore_errors=True)
