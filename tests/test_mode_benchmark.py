"""
tests/test_mode_benchmark.py

Unit tests for Phase 8 Real Processing Mode Benchmarking:
- Schema validation and Safeguard 7 non-simulation scope banner
- ModeBenchmarkEngine sample evaluation across FAST, BALANCED, HIGH
- Analytics benchmark router endpoints (POST /analytics/benchmark/run, GET /analytics/benchmark/{session_id})
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.analytics.mode_benchmark import (
    ModeBenchmarkEngine,
    ModeMetric,
    SampleBenchmarkReport,
)
from main import app

client = TestClient(app)


def test_mode_metric_and_report_schema():
    metric = ModeMetric(
        mode="balanced",
        measured_duration_sec=2.45,
        measured_fps=36.7,
        frames_evaluated=90,
        peak_ram_mb=850.0,
        peak_vram_mb=1200.0,
        file_size_mb=1.85,
        identity_similarity=0.88,
        sharpness_score=86.0,
        resolution="1280x720",
        bitrate="6M",
        blending_strategy="FeatheredBlend",
    )
    assert metric.mode == "balanced"
    assert metric.measured_fps == 36.7
    assert metric.measured_duration_sec > 0

    report = SampleBenchmarkReport(
        benchmark_id="bench_test123",
        session_id="sess_test123",
        hardware_device="GPU (CUDAExecutionProvider)",
        sample_duration_sec=3.5,
        modes={"balanced": metric},
        recommended_mode="balanced",
        recommendation_reason="Balanced mode achieves 36.7 FPS at 720p.",
    )
    data = report.model_dump()
    assert "Benchmark Results: Measured across a representative sample segment" in data["scope_banner"]
    assert data["recommended_mode"] == "balanced"


def test_mode_benchmark_engine_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        img_path = out_dir / "source.jpg"
        img_path.write_bytes(b"dummy image")
        vid_path = out_dir / "target.mp4"
        vid_path.write_bytes(b"dummy video")

        mock_swapper = MagicMock()
        mock_swapper.get_execution_provider.return_value = "CPU (CPUExecutionProvider)"
        mock_swapper.get_source_face.return_value = {"bbox": [10, 10, 100, 100]}
        mock_swapper.process_video_optimized.return_value = (50, 0)

        with patch("backend.app.analytics.mode_benchmark.get_video_info") as mock_info, \
             patch("backend.app.analytics.mode_benchmark.get_file_size_mb", return_value=1.5):

            mock_info.return_value = {
                "fps": 30.0,
                "width": 1280,
                "height": 720,
                "total_frames": 100,
            }

            report = ModeBenchmarkEngine.run_sample_benchmark(
                swapper=mock_swapper,
                img_path=str(img_path),
                vid_path=str(vid_path),
                session_id="session_abc12345",
                output_dir=out_dir,
                sample_seconds=3.0,
            )

            assert report.session_id == "session_abc12345"
            assert "fast" in report.modes
            assert "balanced" in report.modes
            assert "high" in report.modes
            assert report.modes["fast"].frames_evaluated == 90
            assert report.modes["balanced"].resolution == "1280x720"
            assert report.recommended_mode in ["fast", "balanced", "high"]
            assert len(report.recommendation_reason) > 10


def test_benchmark_router_endpoints():
    # Non-existent session
    res_404 = client.post("/analytics/benchmark/run?session_id=non_existent_sess_123")
    assert res_404.status_code == 404

    res_get_404 = client.get("/analytics/benchmark/non_existent_sess_123")
    assert res_get_404.status_code == 404

    # Existing cached report retrieval
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_sess_id = f"test_{uuid.uuid4().hex[:8]}"
    sample_json = reports_dir / f"benchmark_report_{test_sess_id}.json"
    sample_json.write_text(json.dumps({
        "benchmark_id": f"bench_{test_sess_id}",
        "session_id": test_sess_id,
        "scope_banner": "Benchmark Results: Measured across a representative sample segment (non-simulated).",
        "recommended_mode": "balanced",
        "modes": {}
    }), encoding="utf-8")

    res_get_ok = client.get(f"/analytics/benchmark/{test_sess_id}")
    assert res_get_ok.status_code == 200
    data = res_get_ok.json()
    assert data["session_id"] == test_sess_id
    assert data["recommended_mode"] == "balanced"
    sample_json.unlink(missing_ok=True)
