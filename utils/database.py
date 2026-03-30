import sqlite3
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

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
                    created_at TEXT
                )
            """)
            conn.commit()

    def insert_job(self, job_data: Dict[str, Any]):
        keys = list(job_data.keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        values = tuple(job_data[k] for k in keys)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", values)
            conn.commit()

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        if not updates:
            return
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = tuple(updates.values()) + (job_id,)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recent_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_running_job(self) -> Optional[Dict[str, Any]]:
        """Find any job that was left in 'running' or 'queued' state (recovery)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM jobs WHERE status IN ('running', 'queued') ORDER BY created_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def fail_stalled_jobs(self):
        """Mark any 'running' or 'queued' jobs as failed (crash recovery)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE jobs SET status = 'error', stage = 'error', message = 'Job stalled during system restart' "
                "WHERE status IN ('running', 'queued')"
            )
            count = cursor.rowcount
            if count:
                logger.info("Marked %d stalled jobs as failed.", count)
            conn.commit()
