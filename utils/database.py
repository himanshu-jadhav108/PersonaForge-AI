import logging
import sqlite3
from typing import Any

logger = logging.getLogger("personaforge.database")


class JobDB:
    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    kind TEXT,
                    status TEXT,
                    stage TEXT,
                    progress INTEGER,
                    message TEXT,
                    output TEXT,
                    file_size_mb REAL,
                    device TEXT,
                    mode TEXT,
                    similarity_score REAL,
                    orientation TEXT,
                    input_width INTEGER,
                    input_height INTEGER,
                    resize_mode TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    processing_time_sec REAL
                )
            """)

            # Migration for existing DBs
            try:
                cursor.execute("ALTER TABLE jobs ADD COLUMN completed_at TEXT")
            except sqlite3.OperationalError:
                pass

            try:
                cursor.execute("ALTER TABLE jobs ADD COLUMN processing_time_sec REAL")
            except sqlite3.OperationalError:
                pass

            conn.commit()

    def insert_job(self, job_data: dict[str, Any]):
        keys = list(job_data.keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        values = tuple(job_data[k] for k in keys)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", values)
            conn.commit()

    def update_job(self, job_id: str, updates: dict[str, Any]):
        if not updates:
            return
        set_clause = ", ".join([f"{k} = ?" for k in updates])
        values = (*updates.values(), job_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_running_job(self) -> dict[str, Any] | None:
        """Find any job that was left in an active processing or queued state."""
        active_statuses = ("running", "queued", "analyzing", "processing", "validating", "encoding")
        placeholders = ", ".join(["?"] * len(active_statuses))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at DESC LIMIT 1",
                active_statuses,
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def fail_stalled_jobs(self):
        """Mark any active or queued jobs as failed (crash recovery)."""
        active_statuses = ("running", "queued", "analyzing", "processing", "validating", "encoding")
        placeholders = ", ".join(["?"] * len(active_statuses))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE jobs SET status = 'error', stage = 'error', message = 'Job stalled during system restart' "
                f"WHERE status IN ({placeholders})",
                active_statuses,
            )
            count = cursor.rowcount
            if count:
                logger.info("Marked %d stalled jobs as failed.", count)
            conn.commit()

    def delete_jobs_older_than(self, cutoff_iso: str) -> int:
        """Delete jobs created before the given ISO timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff_iso,))
            count = cursor.rowcount
            conn.commit()
            return count

    def delete_job(self, job_id: str) -> bool:
        """Delete a single job by id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            count = cursor.rowcount
            conn.commit()
            return count > 0

    def count_jobs_by_status(self) -> dict[str, int]:
        """Return a mapping of status -> count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
            return {row[0]: row[1] for row in cursor.fetchall()}
