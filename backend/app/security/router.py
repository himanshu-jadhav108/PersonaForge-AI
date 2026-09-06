"""
backend/app/security/router.py — System health, retention telemetry, and storage cleanup endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from models.model_manager import check_models

from .retention import RetentionCleanupReport, RetentionManager, RetentionStats

router = APIRouter(prefix="/system", tags=["Security & System"])
_retention_mgr = RetentionManager()


@router.get("/retention", response_model=RetentionStats, summary="Retrieve storage telemetry and retention policies")
async def get_retention_stats():
    """Returns disk usage across uploads, outputs, temporary frame buffers, and database job distribution."""
    try:
        stats = _retention_mgr.get_storage_stats()
        return JSONResponse(stats.model_dump())
    except Exception as exc:
        raise HTTPException(500, f"Error computing retention statistics: {exc!s}") from exc


@router.post("/cleanup", response_model=RetentionCleanupReport, summary="Trigger storage and data retention cleanup")
async def trigger_retention_cleanup(
    retention_hours: float | None = Query(None, ge=0.0, description="Override retention threshold in hours (default: configured RETENTION_HOURS)"),
    force_all_temp: bool = Query(False, description="Whether to purge all temporary frames regardless of age"),
):
    """Purges expired uploads, outputs, temporary frames, and old SQLite job records."""
    try:
        report = _retention_mgr.perform_cleanup(
            retention_hours=retention_hours,
            force_all_temp=force_all_temp,
        )
        return JSONResponse(report.model_dump())
    except Exception as exc:
        raise HTTPException(500, f"Error executing retention cleanup: {exc!s}") from exc


@router.get("/health", summary="System health and hardware status check")
async def get_system_health():
    """Reports system status, model asset presence, and environment capabilities."""
    models_ok = False
    model_err = None
    try:
        check_models(auto_download=False)
        models_ok = True
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        model_err = str(exc)

    return JSONResponse({
        "status": "healthy" if models_ok else "degraded",
        "models_verified": models_ok,
        "models_error": model_err,
        "retention_hours": _retention_mgr.retention_hours,
    })
