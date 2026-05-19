"""Cron Job Manager for Jarvis Assistant.

Provides a visual interface and API for creating/editing/deleting scheduled tasks.
Goes beyond notes-with-commands by supporting full cron expressions, job history,
enable/disable, and output capture.

Jobs are stored in SQLite and evaluated by the scheduler every minute.
"""

import json
import logging
import sqlite3
import subprocess
import sys
import platform
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class CronManager:
    """Manages scheduled cron-like jobs.

    Attributes:
        db_manager: Database manager for persistent storage.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cron_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    command TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    username TEXT,
                    description TEXT,
                    last_run DATETIME,
                    last_status TEXT,
                    last_output TEXT,
                    run_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS cron_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    finished_at DATETIME,
                    exit_code INTEGER,
                    output TEXT,
                    error TEXT,
                    FOREIGN KEY (job_id) REFERENCES cron_jobs(id)
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init cron tables: %s", e)

    def create_job(self, name: str, command: str, schedule: str,
                   description: str = "", username: str = None) -> dict:
        """Create a new cron job.

        Args:
            name: Human-readable job name.
            command: Shell command to execute.
            schedule: Cron expression or simple pattern.
                Simple: "every 5m", "every 1h", "daily 08:00", "hourly", "weekly mon 09:00"
                Cron: "*/5 * * * *" (every 5 min), "0 8 * * 1-5" (weekdays 8am)
            description: Optional description.
            username: Owner.

        Returns:
            Dict with job ID or error.
        """
        if not name or not command or not schedule:
            return {"error": "name, command, and schedule are required"}

        try:
            conn = self._get_conn()
            cursor = conn.execute("""
                INSERT INTO cron_jobs (name, command, schedule, description, username)
                VALUES (?, ?, ?, ?, ?)
            """, (name, command, schedule, description, username))
            conn.commit()
            job_id = cursor.lastrowid
            conn.close()
            return {"id": job_id, "message": f"Cron job '{name}' created."}
        except Exception as e:
            return {"error": str(e)}

    def list_jobs(self, username: str = None) -> list[dict]:
        """List all cron jobs."""
        try:
            conn = self._get_conn()
            if username:
                cursor = conn.execute("SELECT * FROM cron_jobs WHERE username = ? ORDER BY created_at DESC", (username,))
            else:
                cursor = conn.execute("SELECT * FROM cron_jobs ORDER BY created_at DESC")
            jobs = []
            for row in cursor.fetchall():
                jobs.append({
                    "id": row["id"], "name": row["name"], "command": row["command"],
                    "schedule": row["schedule"], "enabled": bool(row["enabled"]),
                    "description": row["description"], "username": row["username"],
                    "last_run": row["last_run"], "last_status": row["last_status"],
                    "last_output": (row["last_output"] or "")[:200],
                    "run_count": row["run_count"], "created_at": row["created_at"],
                })
            conn.close()
            return jobs
        except Exception as e:
            logger.error("Failed to list cron jobs: %s", e)
            return []

    def get_job(self, job_id: int) -> Optional[dict]:
        try:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM cron_jobs WHERE id = ?", (job_id,)).fetchone()
            conn.close()
            if not row:
                return None
            return {
                "id": row["id"], "name": row["name"], "command": row["command"],
                "schedule": row["schedule"], "enabled": bool(row["enabled"]),
                "description": row["description"], "username": row["username"],
                "last_run": row["last_run"], "last_status": row["last_status"],
                "last_output": row["last_output"], "run_count": row["run_count"],
                "created_at": row["created_at"],
            }
        except Exception:
            return None

    def update_job(self, job_id: int, updates: dict) -> dict:
        allowed = {"name", "command", "schedule", "enabled", "description"}
        try:
            conn = self._get_conn()
            clauses, values = [], []
            for k, v in updates.items():
                if k in allowed:
                    clauses.append(f"{k} = ?")
                    values.append(v)
            if not clauses:
                conn.close()
                return {"error": "No valid fields"}
            values.append(job_id)
            conn.execute(f"UPDATE cron_jobs SET {', '.join(clauses)} WHERE id = ?", values)
            conn.commit()
            conn.close()
            return {"message": "Job updated."}
        except Exception as e:
            return {"error": str(e)}

    def delete_job(self, job_id: int) -> dict:
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM cron_history WHERE job_id = ?", (job_id,))
            cursor = conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
            conn.commit()
            conn.close()
            return {"message": "Job deleted."} if cursor.rowcount else {"error": "Not found"}
        except Exception as e:
            return {"error": str(e)}

    def get_history(self, job_id: int, limit: int = 20) -> list[dict]:
        try:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT * FROM cron_history WHERE job_id = ? ORDER BY started_at DESC LIMIT ?",
                (job_id, limit))
            results = [{
                "id": r["id"], "job_id": r["job_id"], "started_at": r["started_at"],
                "finished_at": r["finished_at"], "exit_code": r["exit_code"],
                "output": (r["output"] or "")[:500], "error": (r["error"] or "")[:500],
            } for r in cursor.fetchall()]
            conn.close()
            return results
        except Exception:
            return []

    def run_job(self, job_id: int) -> dict:
        """Manually trigger a job."""
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        output, error, exit_code = self._execute_command(job["command"])
        self._record_execution(job_id, output, error, exit_code)
        return {"output": output[:500], "error": error[:500], "exit_code": exit_code}

    def evaluate_jobs(self) -> None:
        """Check all enabled jobs and run those whose schedule matches now. Called every minute."""
        now = datetime.now()
        jobs = self.list_jobs()
        for job in jobs:
            if not job["enabled"]:
                continue
            if self._matches_schedule(now, job["schedule"], job.get("last_run")):
                logger.info("Cron job '%s' triggered (schedule: %s)", job["name"], job["schedule"])
                output, error, exit_code = self._execute_command(job["command"])
                self._record_execution(job["id"], output, error, exit_code)

    def _matches_schedule(self, now: datetime, schedule: str, last_run: str = None) -> bool:
        """Check if current time matches the schedule."""
        import re
        s = schedule.strip().lower()

        # Simple patterns
        if s == "hourly":
            return now.minute == 0
        if s == "daily" or s.startswith("daily "):
            time_part = s[6:].strip() if " " in s else "00:00"
            return now.strftime("%H:%M") == time_part
        if s.startswith("weekly"):
            parts = s.split()
            if len(parts) >= 2:
                days_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
                day = days_map.get(parts[1][:3], -1)
                if now.weekday() != day:
                    return False
                time_part = parts[2] if len(parts) > 2 else "00:00"
                return now.strftime("%H:%M") == time_part
            return now.weekday() == 0 and now.hour == 0 and now.minute == 0

        # Interval: "every Nm" / "every Nh"
        interval_match = re.match(r"every\s+(\d+)\s*([mhd])", s)
        if interval_match:
            amount = int(interval_match.group(1))
            unit = interval_match.group(2)
            if unit == "m":
                return now.minute % amount == 0
            elif unit == "h":
                return now.minute == 0 and now.hour % amount == 0
            elif unit == "d":
                return now.hour == 0 and now.minute == 0
            return False

        # Cron expression: "m h dom mon dow"
        parts = s.split()
        if len(parts) == 5:
            return self._match_cron(now, parts)

        return False

    def _match_cron(self, now: datetime, parts: list) -> bool:
        """Match a 5-field cron expression."""
        fields = [now.minute, now.hour, now.day, now.month, now.weekday()]
        ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]

        for i, (field_val, cron_part) in enumerate(zip(fields, parts)):
            if not self._cron_field_matches(field_val, cron_part, ranges[i]):
                return False
        return True

    def _cron_field_matches(self, value: int, pattern: str, valid_range: tuple) -> bool:
        """Check if a value matches a cron field pattern."""
        if pattern == "*":
            return True
        # */N
        if pattern.startswith("*/"):
            try:
                step = int(pattern[2:])
                return value % step == 0
            except ValueError:
                return False
        # Comma-separated
        if "," in pattern:
            return value in [int(x) for x in pattern.split(",") if x.isdigit()]
        # Range: N-M
        if "-" in pattern:
            parts = pattern.split("-")
            try:
                return int(parts[0]) <= value <= int(parts[1])
            except (ValueError, IndexError):
                return False
        # Exact
        try:
            return value == int(pattern)
        except ValueError:
            return False

    def _execute_command(self, command: str) -> tuple:
        """Execute a command and return (stdout, stderr, exit_code)."""
        if command.startswith("python ") or command.startswith("python3 "):
            prefix = "python3 " if command.startswith("python3 ") else "python "
            script_part = command[len(prefix):]
            command = f'"{sys.executable}" {script_part}'

        try:
            if platform.system() == "Windows":
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
            else:
                result = subprocess.run(command, shell=True, executable="/bin/bash",
                                        capture_output=True, text=True, timeout=120)
            return (result.stdout or "", result.stderr or "", result.returncode)
        except subprocess.TimeoutExpired:
            return ("", "Timed out after 120s", -1)
        except Exception as e:
            return ("", str(e), -1)

    def _record_execution(self, job_id: int, output: str, error: str, exit_code: int):
        """Record job execution in history and update job stats."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "success" if exit_code == 0 else f"failed (exit {exit_code})"
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO cron_history (job_id, finished_at, exit_code, output, error)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, now_str, exit_code, output[:2000], error[:2000]))
            conn.execute("""
                UPDATE cron_jobs SET last_run = ?, last_status = ?, last_output = ?,
                       run_count = run_count + 1 WHERE id = ?
            """, (now_str, status, (output or error)[:500], job_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to record cron execution: %s", e)
