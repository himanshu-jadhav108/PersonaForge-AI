import time
import psutil
import threading
from pathlib import Path
from typing import List
import uuid
import asyncio

from benchmark.models import BenchmarkConfig, BenchmarkResult
from benchmark.data_gen import generate_dummy_video, generate_dummy_source_face
from face_swap import FaceSwapper
from backend.app.identity.validator import IdentityValidator
from backend.app.quality.assessor import FaceQualityAssessor

class BenchmarkRunner:
    def __init__(self):
        self.swapper = FaceSwapper() # Initialize global swapper for benchmarks
        self.swapper._warm_up()
        
    def track_resources(self, stop_event: threading.Event, stats: dict):
        """Background thread to track peak RAM and VRAM during a run."""
        peak_ram = 0
        peak_vram = 0
        
        has_gputil = False
        try:
            import GPUtil
            has_gputil = True
        except ImportError:
            pass
            
        while not stop_event.is_set():
            mem = psutil.virtual_memory()
            used_ram_mb = mem.used / (1024 * 1024)
            if used_ram_mb > peak_ram:
                peak_ram = used_ram_mb
                
            if has_gputil:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    used_vram_mb = gpus[0].memoryUsed
                    if used_vram_mb > peak_vram:
                        peak_vram = used_vram_mb
                        
            time.sleep(0.1)
            
        stats['peak_ram_mb'] = peak_ram
        stats['peak_vram_mb'] = peak_vram

    def run_benchmark(self, config: BenchmarkConfig, output_dir: Path) -> BenchmarkResult:
        print(f"Running benchmark {config.id}: {config.resolution}, {config.device}, Faces: {config.face_count}")
        
        # 1. Generate data
        vid_path = output_dir / f"{config.id}_vid.mp4"
        img_path = output_dir / f"{config.id}_src.jpg"
        
        generate_dummy_video(vid_path, config.resolution, config.face_count, config.frames)
        generate_dummy_source_face(img_path)
        
        # 2. Setup tracking
        stop_event = threading.Event()
        stats = {}
        tracker_thread = threading.Thread(target=self.track_resources, args=(stop_event, stats))
        tracker_thread.start()
        
        # 3. Execute Swapping Logic (Simulated API for benchmark)
        start_time = time.perf_counter()
        
        # Since this is an automated benchmark, we bypass the async API queue and hit FaceSwapper directly
        # For multi-face/batch size, we simulate the workload scaling
        job_id = f"bench_{uuid.uuid4().hex[:8]}"
        frames_dir = output_dir / f"frames_{job_id}"
        processed_dir = output_dir / f"processed_{job_id}"
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        from video_utils import extract_frames
        fps, total_frames, _ = extract_frames(str(vid_path), str(frames_dir), None)
        
        source_face = self.swapper.get_source_face(str(img_path))
        
        identity_validator = IdentityValidator(job_id=job_id)
        
        # Note: FaceSwapper currently doesn't expose native batching. 
        # We process frame by frame, but scale the tracking overhead for multi-face.
        class MockDB:
            def update_job(self, *args, **kwargs): pass
            
        from face_swap import QualityMode
        qmode = QualityMode("fast") # Benchmark baseline
        
        swapped, skipped = self.swapper.process_video_optimized(
            source_face, str(frames_dir), str(processed_dir), qmode, -1, None, 0, 100, MockDB(), job_id, identity_validator
        )
        
        end_time = time.perf_counter()
        
        # 4. Stop tracking
        stop_event.set()
        tracker_thread.join()
        
        processing_time = end_time - start_time
        
        # 5. Quality & Identity Scores
        avg_id_score = 0.0
        report = identity_validator.generate_report()
        if report:
            avg_id_score = report.average_similarity * 100.0
            
        # Sample quality score from first output frame
        out_frames = list(processed_dir.glob("*.jpg"))
        avg_quality_score = 0.0
        if out_frames:
            assessor = FaceQualityAssessor(face_analysis_app=self.swapper._app)
            try:
                q_report = assessor.assess_image(str(out_frames[0]))
                avg_quality_score = q_report.quality_score
            except:
                pass
                
        # Clean up
        vid_path.unlink(missing_ok=True)
        img_path.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
        
        return BenchmarkResult(
            config_id=config.id,
            processing_time_sec=round(processing_time, 2),
            fps=round(total_frames / processing_time, 2) if processing_time > 0 else 0.0,
            peak_ram_mb=round(stats.get('peak_ram_mb', 0), 2),
            peak_vram_mb=round(stats.get('peak_vram_mb', 0), 2),
            avg_identity_score=round(avg_id_score, 2),
            avg_quality_score=round(avg_quality_score, 2)
        )

if __name__ == "__main__":
    from benchmark.report import generate_csv_report
    from benchmark.charts import generate_charts
    import sys
    
    output_dir = Path("outputs/benchmark_temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = [
        BenchmarkConfig(id="cpu_720p_single", device="cpu", resolution="720p", face_count=1, frames=10),
        BenchmarkConfig(id="cpu_1080p_single", device="cpu", resolution="1080p", face_count=1, frames=10),
        BenchmarkConfig(id="cpu_720p_multi", device="cpu", resolution="720p", face_count=3, frames=10),
    ]
    
    runner = BenchmarkRunner()
    results = []
    for c in configs:
        res = runner.run_benchmark(c, output_dir)
        results.append(res)
        
    generate_csv_report(results, Path("benchmark/reports"))
    generate_charts(results, Path("benchmark/charts"))
    print("Benchmarking complete.")
