"""Persistent priority task queue for the nightly autopilot mode (Phase 5).

User-submitted tasks (added via chat or the API) are given the highest
priority so they're always worked on before anything the agent discovered
on its own. If the queue is empty, app.autopilot.discovery is used instead.
"""

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


class AutopilotTaskQueue:
    """SQLite-backed queue of autopilot tasks.

    Task lifecycle (status column):
        queued -> in_progress -> done
                              -> rolled_back
                              -> needs_clarification   (ask_user during an unattended run)
                              -> skipped                (no verification method / config error)
                              -> awaiting_user_confirmation -> confirmed_done
                                                            -> rolled_back
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        try:
            conn = self._get_conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS autopilot_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    priority INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'queued',
                    session_id TEXT,
                    verify_command TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init autopilot_tasks table: %s", e)

    def add_task(
        self,
        task: str,
        source: str = "user",
        verify_command: Optional[str] = None,
    ) -> int:
        """Add a task to the queue. User-submitted tasks jump to the front."""
        conn = self._get_conn()
        priority = 0
        if source == "user":
            row = conn.execute("SELECT MAX(priority) AS m FROM autopilot_tasks").fetchone()
            priority = (row["m"] or 0) + 1
        cursor = conn.execute(
            "INSERT INTO autopilot_tasks (task, source, priority, verify_command) VALUES (?, ?, ?, ?)",
            (task, source, priority, verify_command),
        )
        conn.commit()
        task_id = cursor.lastrowid
        conn.close()
        logger.info("Queued autopilot task #%d (source=%s): %s", task_id, source, task[:80])
        return task_id

    def next_queued(self) -> Optional[dict]:
        """Return the highest-priority queued task, or None if the queue is empty."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM autopilot_tasks WHERE status = 'queued' "
            "ORDER BY priority DESC, created_at ASC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get(self, task_id: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM autopilot_tasks WHERE id = ?", (task_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_tasks(self, status: Optional[str] = None) -> list[dict]:
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM autopilot_tasks WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM autopilot_tasks ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_status(self, task_id: int, status: str, session_id: Optional[str] = None, notes: Optional[str] = None) -> None:
        """Update a task's status and optional session_id/notes."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE autopilot_tasks SET status = ?, session_id = COALESCE(?, session_id), "
            "notes = COALESCE(?, notes), updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, session_id, notes, task_id),
        )
        conn.commit()
        conn.close()
