"""
backend/app/security/retention.py — Data retention, disk usage telemetry, and scheduled artifact purging.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from utils.database import JobDB

logger = logging.getLogger("personaforge.security.retention")


class StorageCategoryStats(BaseModel):
    file_count: int = Field(0, description="Total number of files in category")
    size_mb: float = Field(0.0, description="Total size in megabytes")


class RetentionStats(BaseModel):
    retention_hours: float = Field(..., description="Active retention threshold in hours")
    uploads: StorageCategoryStats = Field(default_factory=StorageCategoryStats)
    outputs: StorageCategoryStats = Field(default_factory=StorageCategoryStats)
    temp_frames: StorageCategoryStats = Field(default_factory=StorageCategoryStats)
    total_storage_mb: float = Field(0.0, description="Total monitored storage in MB")
    jobs_by_status: dict[str, int] = Field(default_factory=dict, description="Job counts broken down by status")
    stalled_jobs: int = Field(0, description="Number of potentially stalled jobs")


class RetentionCleanupReport(BaseModel):
    retention_hours_used: float = Field(..., description="Retention threshold applied for cleanup")
    deleted_uploads: int = Field(0, description="Number of upload files or directories removed")
    deleted_outputs: int = Field(0, description="Number of output video/image files removed")
    deleted_frames: int = Field(0, description="Number of temporary frame directories removed")
    deleted_jobs: int = Field(0, description="Number of database job records purged")
    freed_bytes: int = Field(0, description="Total disk space freed in bytes")
    freed_mb: float = Field(0.0, description="Total disk space freed in megabytes")
    duration_sec: float = Field(0.0, description="Duration of cleanup operation in seconds")
    timestamp: str = Field(..., description="ISO 8601 completion timestamp")


class RetentionManager:
    """Orchestrates automated and on-demand privacy and storage lifecycle policies."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        retention_hours: float | None = None,
        db: JobDB | None = None,
    ):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parents[3]
        else:
            self.base_dir = Path(base_dir)

        if retention_hours is None:
            try:
                self.retention_hours = float(os.getenv("RETENTION_HOURS", "24"))
            except ValueError:
                self.retention_hours = 24.0
        else:
            self.retention_hours = float(retention_hours)

        self.uploads_dir = self.base_dir / "uploads"
        self.frames_dir = self.base_dir / "temp_frames"
        self.outputs_dir = self.base_dir / "outputs"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.db = db or JobDB(str(self.base_dir / "jobs.db"))

    def _dir_stats(self, directory: Path) -> StorageCategoryStats:
        count = 0
        total_size = 0
        if directory.exists():
            for p in directory.rglob("*"):
                if p.is_file():
                    count += 1
                    try:
                        total_size += p.stat().st_size
                    except (OSError, PermissionError):
                        pass
        return StorageCategoryStats(
            file_count=count,
            size_mb=round(total_size / (1024 * 1024), 2),
        )

    def get_storage_stats(self) -> RetentionStats:
        """Gather disk usage metrics and database job statistics."""
        up_stats = self._dir_stats(self.uploads_dir)
        out_stats = self._dir_stats(self.outputs_dir)
        frm_stats = self._dir_stats(self.frames_dir)

        total_mb = round(up_stats.size_mb + out_stats.size_mb + frm_stats.size_mb, 2)
        status_counts = self.db.count_jobs_by_status()
        stalled = sum(status_counts.get(s, 0) for s in ("queued", "running", "analyzing", "processing", "validating", "encoding"))

        return RetentionStats(
            retention_hours=self.retention_hours,
            uploads=up_stats,
            outputs=out_stats,
            temp_frames=frm_stats,
            total_storage_mb=total_mb,
            jobs_by_status=status_counts,
            stalled_jobs=stalled,
        )

    def perform_cleanup(
        self,
        retention_hours: float | None = None,
        force_all_temp: bool = False,
    ) -> RetentionCleanupReport:
        """Purge temporary frames, aged uploads, outputs, and obsolete DB records."""
        t0 = time.perf_counter()
        active_hours = retention_hours if retention_hours is not None else self.retention_hours
        retention_sec = active_hours * 3600.0
        now = time.time()

        del_uploads = 0
        del_outputs = 0
        del_frames = 0
        freed_bytes = 0

        # 1. Clean temp frames (keep max 1 hour or 0 if forced)
        temp_threshold = 0.0 if force_all_temp else 3600.0
        if self.frames_dir.exists():
            for p in self.frames_dir.iterdir():
                try:
                    mtime = p.stat().st_mtime
                    if (now - mtime) > temp_threshold:
                        size = self._get_path_size(p)
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            p.unlink(missing_ok=True)
                        del_frames += 1
                        freed_bytes += size
                except (OSError, PermissionError) as exc:
                    logger.debug("Could not clean temp frame item %s: %s", p, exc)

        # 2. Clean uploads (session folders or orphaned files)
        if self.uploads_dir.exists():
            for p in self.uploads_dir.iterdir():
                try:
                    mtime = p.stat().st_mtime
                    if (now - mtime) > retention_sec:
                        size = self._get_path_size(p)
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            p.unlink(missing_ok=True)
                        del_uploads += 1
                        freed_bytes += size
                except (OSError, PermissionError) as exc:
                    logger.debug("Could not clean upload item %s: %s", p, exc)

        # 3. Clean outputs (videos, reports, thumbnails older than retention)
        if self.outputs_dir.exists():
            for p in self.outputs_dir.iterdir():
                try:
                    # If directory (e.g. reports, selection_thumbnails), check items inside
                    if p.is_dir():
                        for sub_p in p.iterdir():
                            mtime = sub_p.stat().st_mtime
                            if (now - mtime) > retention_sec:
                                size = self._get_path_size(sub_p)
                                sub_p.unlink(missing_ok=True)
                                del_outputs += 1
                                freed_bytes += size
                    elif p.is_file():
                        mtime = p.stat().st_mtime
                        if (now - mtime) > retention_sec:
                            size = p.stat().st_size
                            p.unlink(missing_ok=True)
                            del_outputs += 1
                            freed_bytes += size
                except (OSError, PermissionError) as exc:
                    logger.debug("Could not clean output item %s: %s", p, exc)

        # 4. Clean old DB jobs
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=active_hours)
        cutoff_iso = cutoff_dt.isoformat()
        del_jobs = self.db.delete_jobs_older_than(cutoff_iso)

        duration = time.perf_counter() - t0
        report = RetentionCleanupReport(
            retention_hours_used=active_hours,
            deleted_uploads=del_uploads,
            deleted_outputs=del_outputs,
            deleted_frames=del_frames,
            deleted_jobs=del_jobs,
            freed_bytes=freed_bytes,
            freed_mb=round(freed_bytes / (1024 * 1024), 2),
            duration_sec=round(duration, 3),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "Retention cleanup completed in %.3fs: freed %.2fMB across %d uploads, %d outputs, %d frame sets, %d jobs.",
            duration,
            report.freed_mb,
            del_uploads,
            del_outputs,
            del_frames,
            del_jobs,
        )
        return report

    @staticmethod
    def _get_path_size(path: Path) -> int:
        if path.is_file():
            try:
                return path.stat().st_size
            except (OSError, PermissionError):
                return 0
        total = 0
        if path.is_dir():
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except (OSError, PermissionError):
                        pass
        return total
