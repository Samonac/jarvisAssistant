"""Log Analyzer for Jarvis Assistant.

Tails log files, detects anomalies (error spikes, new error patterns),
and alerts the user via notifications.

Features:
- Watch multiple log files simultaneously
- Detect error rate spikes (compared to baseline)
- Pattern matching for known error signatures
- Configurable alert thresholds
- Log search and tail via API
"""

import logging
import os
import re
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Common error patterns across log formats
DEFAULT_ERROR_PATTERNS = [
    r"\b(ERROR|CRITICAL|FATAL|EXCEPTION)\b",
    r"\b(Traceback|traceback)\b",
    r"\b(FAILED|failed|failure)\b",
    r"\b(OOM|OutOfMemory|MemoryError)\b",
    r"\b(ConnectionRefused|ConnectionReset|TimeoutError)\b",
    r"\b(PermissionDenied|AccessDenied|Unauthorized)\b",
    r"\b(disk full|no space left)\b",
    r"HTTP\s+[45]\d{2}\b",
]


class LogAnalyzer:
    """Monitors log files and detects anomalies.

    Attributes:
        db_manager: Database manager for persistent storage.
        scheduler: Reference to scheduler for pushing alerts.
        watched_files: Dict of file paths being monitored.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.scheduler = None
        self._watched_files: dict = {}  # path -> {position, error_count, last_check}
        self._error_window: dict = {}  # path -> deque of (timestamp, line)
        self._alert_cooldown: dict = {}  # path -> last_alert_time
        self._lock = threading.Lock()
        self._error_patterns = [re.compile(p, re.IGNORECASE) for p in DEFAULT_ERROR_PATTERNS]

        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS log_watches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    label TEXT,
                    enabled INTEGER DEFAULT 1,
                    error_threshold INTEGER DEFAULT 5,
                    window_minutes INTEGER DEFAULT 5,
                    username TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS log_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    error_count INTEGER,
                    sample_lines TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init log analyzer tables: %s", e)

    # ── Watch Management ──────────────────────────────────────────────────

    def add_watch(self, file_path: str, label: str = None,
                  error_threshold: int = 5, window_minutes: int = 5,
                  username: str = None) -> dict:
        """Add a log file to watch."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO log_watches (file_path, label, error_threshold, window_minutes, username)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, label or os.path.basename(file_path), error_threshold, window_minutes, username))
            conn.commit()
            conn.close()
            # Initialize tracking
            self._watched_files[file_path] = {
                "position": os.path.getsize(file_path),
                "error_count": 0,
                "last_check": datetime.now(),
            }
            self._error_window[file_path] = deque(maxlen=100)
            return {"message": f"Now watching: {file_path}"}
        except Exception as e:
            return {"error": str(e)}

    def remove_watch(self, file_path: str) -> dict:
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM log_watches WHERE file_path = ?", (file_path,))
            conn.commit()
            conn.close()
            self._watched_files.pop(file_path, None)
            self._error_window.pop(file_path, None)
            return {"message": "Watch removed."}
        except Exception as e:
            return {"error": str(e)}

    def list_watches(self) -> list[dict]:
        try:
            conn = self._get_conn()
            cursor = conn.execute("SELECT * FROM log_watches ORDER BY created_at DESC")
            results = [{
                "id": r["id"], "file_path": r["file_path"], "label": r["label"],
                "enabled": bool(r["enabled"]), "error_threshold": r["error_threshold"],
                "window_minutes": r["window_minutes"], "username": r["username"],
            } for r in cursor.fetchall()]
            conn.close()
            return results
        except Exception:
            return []

    # ── Log Reading ───────────────────────────────────────────────────────

    def tail(self, file_path: str, lines: int = 50) -> dict:
        """Read the last N lines of a log file."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            tail_lines = all_lines[-lines:]
            return {
                "file": file_path,
                "total_lines": len(all_lines),
                "lines": [l.rstrip() for l in tail_lines],
            }
        except Exception as e:
            return {"error": str(e)}

    def search(self, file_path: str, pattern: str, max_results: int = 50) -> dict:
        """Search a log file for lines matching a pattern."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = []
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append({"line_num": i, "content": line.rstrip()[:300]})
                        if len(matches) >= max_results:
                            break
            return {"file": file_path, "pattern": pattern, "matches": matches, "count": len(matches)}
        except re.error as e:
            return {"error": f"Invalid regex: {e}"}
        except Exception as e:
            return {"error": str(e)}

    def get_error_summary(self, file_path: str) -> dict:
        """Get error summary for a watched file."""
        window = self._error_window.get(file_path, deque())
        now = datetime.now()
        recent = [e for e in window if (now - e[0]).total_seconds() < 300]
        return {
            "file": file_path,
            "errors_last_5min": len(recent),
            "total_tracked": len(window),
            "recent_errors": [{"time": e[0].strftime("%H:%M:%S"), "line": e[1][:150]} for e in list(recent)[-5:]],
        }

    # ── Alerts ────────────────────────────────────────────────────────────

    def get_alerts(self, limit: int = 20, unacknowledged_only: bool = False) -> list[dict]:
        try:
            conn = self._get_conn()
            query = "SELECT * FROM log_alerts"
            if unacknowledged_only:
                query += " WHERE acknowledged = 0"
            query += " ORDER BY created_at DESC LIMIT ?"
            cursor = conn.execute(query, (limit,))
            results = [{
                "id": r["id"], "file_path": r["file_path"], "alert_type": r["alert_type"],
                "message": r["message"], "error_count": r["error_count"],
                "sample_lines": r["sample_lines"], "acknowledged": bool(r["acknowledged"]),
                "created_at": r["created_at"],
            } for r in cursor.fetchall()]
            conn.close()
            return results
        except Exception:
            return []

    def acknowledge_alert(self, alert_id: int) -> bool:
        try:
            conn = self._get_conn()
            conn.execute("UPDATE log_alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    # ── Analysis (called by scheduler) ────────────────────────────────────

    def check_logs(self) -> None:
        """Check all watched log files for new errors. Called every ~30 seconds."""
        try:
            conn = self._get_conn()
            cursor = conn.execute("SELECT * FROM log_watches WHERE enabled = 1")
            watches = cursor.fetchall()
            conn.close()
        except Exception:
            return

        for watch in watches:
            file_path = watch["file_path"]
            if not os.path.exists(file_path):
                continue

            threshold = watch["error_threshold"]
            window_min = watch["window_minutes"]

            # Read new lines since last position
            new_lines = self._read_new_lines(file_path)
            if not new_lines:
                continue

            # Check for errors
            errors_found = []
            for line in new_lines:
                if self._is_error_line(line):
                    errors_found.append(line)
                    with self._lock:
                        if file_path not in self._error_window:
                            self._error_window[file_path] = deque(maxlen=100)
                        self._error_window[file_path].append((datetime.now(), line.strip()))

            # Check if error rate exceeds threshold
            if errors_found:
                window = self._error_window.get(file_path, deque())
                cutoff = datetime.now() - timedelta(minutes=window_min)
                recent_count = sum(1 for t, _ in window if t >= cutoff)

                if recent_count >= threshold:
                    self._fire_alert(file_path, recent_count, errors_found[-3:], watch["label"])

    def _read_new_lines(self, file_path: str) -> list[str]:
        """Read lines added since last check."""
        try:
            current_size = os.path.getsize(file_path)
            tracking = self._watched_files.get(file_path)

            if not tracking:
                self._watched_files[file_path] = {"position": current_size, "error_count": 0, "last_check": datetime.now()}
                return []

            last_pos = tracking["position"]

            # File was truncated/rotated
            if current_size < last_pos:
                last_pos = 0

            if current_size == last_pos:
                return []

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                new_lines = f.readlines()

            self._watched_files[file_path]["position"] = current_size
            self._watched_files[file_path]["last_check"] = datetime.now()
            return new_lines

        except Exception as e:
            logger.debug("Error reading log %s: %s", file_path, e)
            return []

    def _is_error_line(self, line: str) -> bool:
        """Check if a line matches any error pattern."""
        return any(p.search(line) for p in self._error_patterns)

    def _fire_alert(self, file_path: str, error_count: int, sample_lines: list, label: str):
        """Fire an alert notification for error spike."""
        # Cooldown: don't alert more than once per 5 minutes per file
        now = datetime.now()
        last_alert = self._alert_cooldown.get(file_path)
        if last_alert and (now - last_alert).total_seconds() < 300:
            return

        self._alert_cooldown[file_path] = now

        samples = "\n".join(l.strip()[:100] for l in sample_lines[-3:])
        message = (
            f"⚠️ Error spike detected in **{label or file_path}**\n"
            f"{error_count} errors in the monitoring window.\n\n"
            f"Recent errors:\n```\n{samples}\n```"
        )

        # Save alert to DB
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO log_alerts (file_path, alert_type, message, error_count, sample_lines)
                VALUES (?, 'error_spike', ?, ?, ?)
            """, (file_path, message, error_count, samples))
            conn.commit()
            conn.close()
        except Exception:
            pass

        # Push notification
        if self.scheduler:
            self.scheduler.notifications.append({
                "message": message,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "note_id": 0,
                "type": "alert",
            })

        logger.warning("Log alert: %d errors in %s", error_count, file_path)
