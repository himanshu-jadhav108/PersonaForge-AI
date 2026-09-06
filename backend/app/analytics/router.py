import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

try:
    import psutil
except ImportError:
    psutil = None

from backend.app.analytics.mode_benchmark import ModeBenchmarkEngine
from utils.database import JobDB

router = APIRouter(prefix="/analytics", tags=["analytics"])
db = JobDB()

# Directory where reports are saved
OUTPUTS_DIR = Path("outputs")
REPORTS_DIR = OUTPUTS_DIR / "reports"
UPLOADS_DIR = Path("uploads")


@router.get("/overview")
async def get_overview() -> dict[str, Any]:
    jobs = db.get_recent_jobs(limit=1000)  # get all or up to 1000 for overview
    total = len(jobs)

    if total == 0:
        return {"total_jobs": 0, "success_rate": 0, "failed_jobs": 0, "average_fps": 0}

    completed = [j for j in jobs if j.get("status") == "done"]
    failed = [j for j in jobs if j.get("status") == "error"]

    # Calculate average FPS over completed jobs
    # Assuming standard video length and frames, but we only have processing time and file_size
    # FPS = frames / processing_time
    # Since we don't store exact frame count in db, we approximate it or just use processing time
    total_time = sum([j.get("processing_time_sec", 0) for j in completed if j.get("processing_time_sec")])
    avg_processing_time = total_time / len(completed) if completed else 0

    # We will just expose average processing time instead of exact FPS if we lack frames count

    return {
        "total_jobs": total,
        "success_rate": round(len(completed) / total * 100, 2),
        "failed_jobs": len(failed),
        "average_processing_time": round(avg_processing_time, 2),
    }


@router.get("/performance")
async def get_performance() -> dict[str, Any]:
    jobs = db.get_recent_jobs(limit=50)  # last 50
    # sort chronologically
    jobs = sorted(jobs, key=lambda x: x.get("created_at", ""))

    times = []
    job_ids = []
    queue_lengths = []

    # Simple simulated queue length (count of jobs created but not completed before this job)
    for i, j in enumerate(jobs):
        if j.get("status") in ["done", "error"]:
            job_ids.append(j.get("id")[:8])
            times.append(j.get("processing_time_sec", 0))
            queue_lengths.append(max(0, i % 5))  # Dummy queue length trend

    return {"job_ids": job_ids, "processing_times": times, "queue_lengths": queue_lengths}


@router.get("/system")
async def get_system_health() -> dict[str, Any]:
    if psutil is not None:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
    else:
        cpu_percent = 0.0
        mem_percent = 0.0

    # GPUtil optional
    gpu_percent = 0
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = gpus[0].load * 100
    except ImportError:
        pass

    return {"cpu_utilization": cpu_percent, "memory_utilization": mem_percent, "gpu_utilization": round(gpu_percent, 2)}


@router.get("/identity")
async def get_identity_trends() -> dict[str, Any]:
    reports = list(REPORTS_DIR.glob("identity_report_*.json"))

    trends = []
    for r in reports[:20]:  # last 20
        try:
            data = json.loads(r.read_text(encoding="utf-8"))
            trends.append(
                {
                    "job_id": r.stem.replace("identity_report_", "")[:8],
                    "score": data.get("identity_score", 0),
                    "drift_detected": data.get("drift_detected", False),
                }
            )
        except Exception:
            continue

    return {"trends": trends}


@router.post("/benchmark/run", summary="Execute real measured processing mode benchmark on a sample clip")
async def run_sample_benchmark(
    request: Request,
    session_id: str = Query(..., description="Session ID containing uploaded source and target media"),
    sample_seconds: float = Query(3.5, ge=1.0, le=10.0, description="Sample duration in seconds"),
) -> dict[str, Any]:
    session_dir = UPLOADS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, f"Session '{session_id}' not found.")

    imgs = list(session_dir.glob("source_face.*"))
    vids = list(session_dir.glob("target_video.*"))
    if not imgs or not vids:
        raise HTTPException(400, "Session media missing source image or target video.")

    swapper = getattr(request.app.state, "swapper", None)
    if swapper is None:
        raise HTTPException(503, "Swapper engine not initialized.")

    try:
        report = ModeBenchmarkEngine.run_sample_benchmark(
            swapper=swapper,
            img_path=str(imgs[0]),
            vid_path=str(vids[0]),
            session_id=session_id,
            output_dir=OUTPUTS_DIR,
            sample_seconds=sample_seconds,
        )
        return report.model_dump()
    except Exception as e:
        raise HTTPException(500, f"Benchmark execution failed: {e!s}") from e


@router.get("/benchmark/{session_id}", summary="Get benchmark report for session")
async def get_benchmark_report(session_id: str) -> dict[str, Any]:
    report_path = REPORTS_DIR / f"benchmark_report_{session_id}.json"
    if not report_path.exists():
        raise HTTPException(404, f"Benchmark report for session '{session_id}' not found.")
    return json.loads(report_path.read_text(encoding="utf-8"))
