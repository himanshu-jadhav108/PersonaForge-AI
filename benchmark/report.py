import csv
import json
from pathlib import Path
from typing import List
from benchmark.models import BenchmarkResult

def generate_csv_report(results: List[BenchmarkResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_history.csv"
    
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "config_id", "timestamp", "processing_time_sec", "fps",
                "peak_ram_mb", "peak_vram_mb", "avg_identity_score", "avg_quality_score"
            ])
            
        for r in results:
            writer.writerow([
                r.config_id, r.timestamp, r.processing_time_sec, r.fps,
                r.peak_ram_mb, r.peak_vram_mb, r.avg_identity_score, r.avg_quality_score
            ])
            
    # Also save the latest run as JSON
    json_path = output_dir / "latest_run.json"
    with open(json_path, 'w') as f:
        json.dump([r.model_dump() for r in results], f, indent=4)
        
    print(f"Reports generated in {output_dir}")
