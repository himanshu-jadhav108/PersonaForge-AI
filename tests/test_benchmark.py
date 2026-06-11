import pytest
from pathlib import Path
from benchmark.models import BenchmarkConfig, BenchmarkResult
from benchmark.report import generate_csv_report
from benchmark.charts import generate_charts
import json
import csv
import tempfile
import os

def test_benchmark_report_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        results = [
            BenchmarkResult(
                config_id="test_run",
                processing_time_sec=10.5,
                fps=24.0,
                peak_ram_mb=1024.0,
                peak_vram_mb=512.0,
                avg_identity_score=95.0,
                avg_quality_score=85.0
            )
        ]
        
        generate_csv_report(results, out_dir)
        
        csv_path = out_dir / "benchmark_history.csv"
        json_path = out_dir / "latest_run.json"
        
        assert csv_path.exists()
        assert json_path.exists()
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            assert len(lines) == 2 # header + 1 row
            assert "test_run" in lines[1]
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["config_id"] == "test_run"

def test_benchmark_charts_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        results = [
            BenchmarkResult(
                config_id="test_run",
                processing_time_sec=10.5,
                fps=24.0,
                peak_ram_mb=1024.0,
                peak_vram_mb=512.0,
                avg_identity_score=95.0,
                avg_quality_score=85.0
            )
        ]
        
        generate_charts(results, out_dir)
        
        assert (out_dir / "processing_time.html").exists()
        assert (out_dir / "fps_comparison.html").exists()
        assert (out_dir / "memory_usage.html").exists()
