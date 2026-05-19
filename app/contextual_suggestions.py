"""Contextual Suggestions Engine for Jarvis Assistant.

Monitors user behavior patterns (commands, tool usage, timing) and proactively
suggests automations, workflows, or optimizations.

Features:
- Tracks all commands/tool calls with timestamps and frequency
- Detects time-based patterns (e.g., "user runs X every morning")
- Detects repetitive commands (same command run N+ times)
- Detects optimizable sequences (multi-step patterns that could be scripted)
- Reads OS command history (bash_history, PowerShell history)
- Generates suggestions as notifications or inline in conversation
"""

import json
import logging
import os
import platform
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class ContextualSuggestions:
    """Monitors user patterns and generates proactive suggestions.

    Attributes:
        db_manager: Database manager for persistent storage.
        scheduler: Reference to scheduler for pushing notifications.
        min_occurrences: Minimum times a pattern must occur before suggesting.
        suggestion_cooldown_hours: Hours before re-suggesting the same thing.
    """

    def __init__(self, db_manager, min_occurrences: int = 3, suggestion_cooldown_hours: int = 24):
        self.db_manager = db_manager
        self.scheduler = None
        self.min_occurrences = min_occurrences
        self.suggestion_cooldown_hours = suggestion_cooldown_hours
        self._lock = threading.Lock()

        self._init_tables()

    def _get_conn(self):
        """Get a thread-safe SQLite connection."""
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create tables for tracking user activity and suggestions."""
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    activity_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_name TEXT,
                    hour_of_day INTEGER,
                    day_of_week INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS suggestions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    suggestion_type TEXT NOT NULL,
                    suggestion_text TEXT NOT NULL,
                    pattern_key TEXT NOT NULL,
                    accepted INTEGER DEFAULT 0,
                    dismissed INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_activity_user ON user_activity(username, timestamp);
                CREATE INDEX IF NOT EXISTS idx_activity_type ON user_activity(activity_type, tool_name);
                CREATE INDEX IF NOT EXISTS idx_activity_hour ON user_activity(hour_of_day, day_of_week);
                CREATE INDEX IF NOT EXISTS idx_suggestions_pattern ON suggestions_log(pattern_key, created_at);
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init suggestion tables: %s", e)

    # ── Activity Tracking ─────────────────────────────────────────────────

    def track_activity(self, username: str, activity_type: str, content: str,
                       tool_name: str = None) -> None:
        """Record a user activity for pattern analysis.

        Args:
            username: Who performed the action.
            activity_type: One of: command, tool_call, chat, code_exec, file_op.
            content: The actual content (command text, tool args, etc.).
            tool_name: Optional tool name if it was a tool call.
        """
        now = datetime.now()
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO user_activity (username, activity_type, content, tool_name,
                                           hour_of_day, day_of_week)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, activity_type, content[:500], tool_name,
                  now.hour, now.weekday()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to track activity: %s", e)

    def track_command(self, username: str, command: str) -> None:
        """Convenience: track a shell command execution."""
        self.track_activity(username, "command", command, tool_name="run_command")

    def track_tool_call(self, username: str, tool_name: str, args: dict) -> None:
        """Convenience: track a tool call."""
        content = f"{tool_name}({json.dumps(args)[:200]})"
        self.track_activity(username, "tool_call", content, tool_name=tool_name)

    # ── OS Command History Ingestion ──────────────────────────────────────

    def ingest_os_history(self, username: str) -> dict:
        """Read the OS shell history and ingest new commands.

        Supports:
        - Linux/macOS: ~/.bash_history, ~/.zsh_history
        - Windows: PowerShell PSReadLine history
        - Windows: cmd.exe doskey history (limited)

        Returns:
            Dict with count of new commands ingested.
        """
        history_commands = []

        if platform.system() == "Windows":
            # PowerShell history file
            ps_history = os.path.join(
                os.environ.get("APPDATA", ""),
                "Microsoft", "Windows", "PowerShell", "PSReadLine",
                "ConsoleHost_history.txt"
            )
            if os.path.exists(ps_history):
                try:
                    with open(ps_history, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    # Take last 200 commands
                    history_commands = [l.strip() for l in lines[-200:] if l.strip()]
                except Exception as e:
                    logger.warning("Failed to read PowerShell history: %s", e)

            # Also try cmd history via doskey (less reliable)
            # doskey /history only works in active session, skip for now

        else:
            # Linux/macOS
            home = os.path.expanduser("~")
            history_files = [
                os.path.join(home, ".bash_history"),
                os.path.join(home, ".zsh_history"),
            ]
            for hf in history_files:
                if os.path.exists(hf):
                    try:
                        with open(hf, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                        # zsh_history has timestamps like ": 1234567890:0;command"
                        for line in lines[-200:]:
                            line = line.strip()
                            if line.startswith(": "):
                                # zsh format
                                parts = line.split(";", 1)
                                if len(parts) > 1:
                                    line = parts[1]
                            if line and not line.startswith("#"):
                                history_commands.append(line)
                    except Exception as e:
                        logger.warning("Failed to read %s: %s", hf, e)
                    break  # Use first found

        if not history_commands:
            return {"ingested": 0, "message": "No shell history found."}

        # Deduplicate against already-ingested commands
        conn = self._get_conn()
        existing = set()
        try:
            cursor = conn.execute(
                "SELECT content FROM user_activity WHERE username = ? AND activity_type = 'os_history' "
                "ORDER BY timestamp DESC LIMIT 500",
                (username,)
            )
            existing = {row["content"] for row in cursor.fetchall()}
        except Exception:
            pass

        new_count = 0
        now = datetime.now()
        for cmd in history_commands:
            if cmd not in existing and len(cmd) > 2:
                try:
                    conn.execute("""
                        INSERT INTO user_activity (username, activity_type, content, tool_name,
                                                   hour_of_day, day_of_week)
                        VALUES (?, 'os_history', ?, NULL, ?, ?)
                    """, (username, cmd[:500], now.hour, now.weekday()))
                    new_count += 1
                except Exception:
                    pass

        conn.commit()
        conn.close()
        logger.info("Ingested %d new OS history commands for user %s", new_count, username)
        return {"ingested": new_count, "total_read": len(history_commands)}

    # ── Pattern Analysis ──────────────────────────────────────────────────

    def analyze_patterns(self, username: str) -> list[dict]:
        """Analyze user activity and generate suggestions.

        Detects:
        1. Time-based patterns (same action at same hour repeatedly)
        2. Repetitive commands (same command run many times)
        3. Sequences (commands that always follow each other)
        4. Optimizable patterns (long commands, piped chains)

        Returns:
            List of suggestion dicts.
        """
        suggestions = []

        try:
            conn = self._get_conn()

            # 1. Repetitive commands (same command 3+ times)
            suggestions.extend(self._detect_repetitive(conn, username))

            # 2. Time-based patterns (same action at same hour on same days)
            suggestions.extend(self._detect_time_patterns(conn, username))

            # 3. Optimizable commands (long, complex, or piped)
            suggestions.extend(self._detect_optimizable(conn, username))

            # 4. Frequent tool usage patterns
            suggestions.extend(self._detect_tool_patterns(conn, username))

            conn.close()
        except Exception as e:
            logger.error("Pattern analysis failed: %s", e)

        # Filter out recently suggested patterns
        suggestions = self._filter_cooldown(suggestions, username)

        return suggestions

    def _detect_repetitive(self, conn, username: str) -> list[dict]:
        """Detect commands run many times."""
        suggestions = []
        # Look at last 7 days
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

        cursor = conn.execute("""
            SELECT content, COUNT(*) as cnt, tool_name
            FROM user_activity
            WHERE username = ? AND timestamp >= ?
              AND activity_type IN ('command', 'os_history', 'tool_call')
            GROUP BY content
            HAVING cnt >= ?
            ORDER BY cnt DESC
            LIMIT 10
        """, (username, cutoff, self.min_occurrences))

        for row in cursor.fetchall():
            content = row["content"]
            count = row["cnt"]
            # Skip very short or trivial commands
            if len(content) < 4 or content.lower() in ("ls", "dir", "cd", "cls", "clear", "pwd"):
                continue

            suggestions.append({
                "type": "repetitive_command",
                "pattern_key": f"repeat:{content[:100]}",
                "title": "Repetitive Command Detected",
                "message": (
                    f"Sir, I've noticed you've run this command {count} times in the past week:\n"
                    f"`{content[:120]}`\n\n"
                    f"Shall I create a workflow to automate this, or wrap it in a scheduled script?"
                ),
                "data": {"command": content, "count": count},
                "actions": ["create_workflow", "create_script", "dismiss"],
            })

        return suggestions

    def _detect_time_patterns(self, conn, username: str) -> list[dict]:
        """Detect actions that happen at the same time regularly."""
        suggestions = []
        cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")

        # Find commands that cluster at specific hours
        cursor = conn.execute("""
            SELECT content, hour_of_day, COUNT(*) as cnt, tool_name
            FROM user_activity
            WHERE username = ? AND timestamp >= ?
              AND activity_type IN ('command', 'tool_call', 'os_history')
            GROUP BY content, hour_of_day
            HAVING cnt >= ?
            ORDER BY cnt DESC
            LIMIT 10
        """, (username, cutoff, self.min_occurrences))

        for row in cursor.fetchall():
            content = row["content"]
            hour = row["hour_of_day"]
            count = row["cnt"]

            if len(content) < 4:
                continue

            hour_str = f"{hour:02d}:00"
            suggestions.append({
                "type": "time_pattern",
                "pattern_key": f"time:{hour}:{content[:80]}",
                "title": "Time-Based Pattern",
                "message": (
                    f"Sir, you typically run this around {hour_str}:\n"
                    f"`{content[:120]}`\n\n"
                    f"I've seen this {count} times. Would you like me to schedule it automatically?"
                ),
                "data": {"command": content, "hour": hour, "count": count},
                "actions": ["create_workflow", "dismiss"],
            })

        return suggestions

    def _detect_optimizable(self, conn, username: str) -> list[dict]:
        """Detect commands that could be optimized (long, piped, complex)."""
        suggestions = []
        cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")

        cursor = conn.execute("""
            SELECT content, COUNT(*) as cnt
            FROM user_activity
            WHERE username = ? AND timestamp >= ?
              AND activity_type IN ('command', 'os_history')
              AND LENGTH(content) > 50
            GROUP BY content
            HAVING cnt >= 2
            ORDER BY LENGTH(content) DESC
            LIMIT 5
        """, (username, cutoff))

        for row in cursor.fetchall():
            content = row["content"]
            count = row["cnt"]

            # Check for pipe chains, multiple commands, or complex flags
            is_complex = (
                "|" in content or
                "&&" in content or
                ";" in content or
                content.count("-") > 3 or
                len(content) > 80
            )

            if is_complex:
                suggestions.append({
                    "type": "optimizable",
                    "pattern_key": f"optimize:{content[:80]}",
                    "title": "Complex Command — Script Candidate",
                    "message": (
                        f"Sir, this complex command has been used {count} times:\n"
                        f"`{content[:150]}`\n\n"
                        f"I could wrap this into a reusable Python script with proper error handling. "
                        f"Shall I create one?"
                    ),
                    "data": {"command": content, "count": count},
                    "actions": ["create_script", "dismiss"],
                })

        return suggestions

    def _detect_tool_patterns(self, conn, username: str) -> list[dict]:
        """Detect frequently used tool combinations."""
        suggestions = []
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

        # Most used tools
        cursor = conn.execute("""
            SELECT tool_name, COUNT(*) as cnt
            FROM user_activity
            WHERE username = ? AND timestamp >= ?
              AND tool_name IS NOT NULL
            GROUP BY tool_name
            ORDER BY cnt DESC
            LIMIT 5
        """, (username, cutoff))

        tool_usage = {row["tool_name"]: row["cnt"] for row in cursor.fetchall()}

        # If weather is checked daily, suggest briefing
        if tool_usage.get("get_weather", 0) >= 5:
            suggestions.append({
                "type": "tool_pattern",
                "pattern_key": "tool:weather_to_briefing",
                "title": "Weather Check Pattern",
                "message": (
                    "Sir, you check the weather quite frequently. "
                    "Would you like me to include it in an automatic daily briefing instead?"
                ),
                "data": {"tool": "get_weather", "count": tool_usage["get_weather"]},
                "actions": ["enable_briefing", "dismiss"],
            })

        # If notes are checked frequently, suggest a morning summary
        if tool_usage.get("get_notes", 0) >= 5:
            suggestions.append({
                "type": "tool_pattern",
                "pattern_key": "tool:notes_summary",
                "title": "Notes Review Pattern",
                "message": (
                    "Sir, you review your notes frequently. "
                    "Shall I add a notes summary to your daily briefing?"
                ),
                "data": {"tool": "get_notes", "count": tool_usage["get_notes"]},
                "actions": ["enable_briefing", "dismiss"],
            })

        return suggestions

    def _filter_cooldown(self, suggestions: list[dict], username: str) -> list[dict]:
        """Remove suggestions that were recently shown (cooldown period)."""
        if not suggestions:
            return []

        cutoff = (datetime.now() - timedelta(hours=self.suggestion_cooldown_hours)).strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT pattern_key FROM suggestions_log WHERE username = ? AND created_at >= ?",
                (username, cutoff)
            )
            recent_keys = {row["pattern_key"] for row in cursor.fetchall()}
            conn.close()

            return [s for s in suggestions if s["pattern_key"] not in recent_keys]
        except Exception:
            return suggestions

    # ── Suggestion Delivery ───────────────────────────────────────────────

    def deliver_suggestions(self, username: str) -> list[dict]:
        """Analyze patterns and deliver suggestions as notifications.

        Called periodically by the scheduler. Returns delivered suggestions.
        """
        suggestions = self.analyze_patterns(username)
        if not suggestions:
            return []

        delivered = []
        conn = self._get_conn()

        for suggestion in suggestions[:3]:  # Max 3 suggestions at a time
            # Log the suggestion
            try:
                conn.execute("""
                    INSERT INTO suggestions_log (username, suggestion_type, suggestion_text, pattern_key)
                    VALUES (?, ?, ?, ?)
                """, (username, suggestion["type"], suggestion["message"][:500],
                      suggestion["pattern_key"]))
            except Exception:
                pass

            # Push as notification
            if self.scheduler:
                self.scheduler.notifications.append({
                    "message": f"💡 **Suggestion**: {suggestion['message']}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note_id": 0,
                    "type": "suggestion",
                    "suggestion_data": suggestion.get("data"),
                })

            delivered.append(suggestion)

        conn.commit()
        conn.close()
        logger.info("Delivered %d suggestions to user %s", len(delivered), username)
        return delivered

    def dismiss_suggestion(self, suggestion_id: int) -> bool:
        """Mark a suggestion as dismissed."""
        try:
            conn = self._get_conn()
            conn.execute(
                "UPDATE suggestions_log SET dismissed = 1 WHERE id = ?", (suggestion_id,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def accept_suggestion(self, suggestion_id: int) -> bool:
        """Mark a suggestion as accepted."""
        try:
            conn = self._get_conn()
            conn.execute(
                "UPDATE suggestions_log SET accepted = 1 WHERE id = ?", (suggestion_id,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_suggestion_history(self, username: str, limit: int = 20) -> list[dict]:
        """Get past suggestions for a user."""
        try:
            conn = self._get_conn()
            cursor = conn.execute("""
                SELECT id, suggestion_type, suggestion_text, pattern_key,
                       accepted, dismissed, created_at
                FROM suggestions_log
                WHERE username = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (username, limit))
            results = [
                {
                    "id": row["id"],
                    "type": row["suggestion_type"],
                    "text": row["suggestion_text"],
                    "pattern_key": row["pattern_key"],
                    "accepted": bool(row["accepted"]),
                    "dismissed": bool(row["dismissed"]),
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            return results
        except Exception:
            return []

    def get_activity_stats(self, username: str) -> dict:
        """Get activity statistics for a user (for the dashboard/settings)."""
        try:
            conn = self._get_conn()
            cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

            # Total activities
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM user_activity WHERE username = ? AND timestamp >= ?",
                (username, cutoff)
            ).fetchone()
            total = row["cnt"]

            # By type
            cursor = conn.execute("""
                SELECT activity_type, COUNT(*) as cnt
                FROM user_activity WHERE username = ? AND timestamp >= ?
                GROUP BY activity_type
            """, (username, cutoff))
            by_type = {row["activity_type"]: row["cnt"] for row in cursor.fetchall()}

            # Top commands
            cursor = conn.execute("""
                SELECT content, COUNT(*) as cnt
                FROM user_activity
                WHERE username = ? AND timestamp >= ?
                  AND activity_type IN ('command', 'os_history')
                GROUP BY content
                ORDER BY cnt DESC
                LIMIT 10
            """, (username, cutoff))
            top_commands = [{"command": row["content"], "count": row["cnt"]} for row in cursor.fetchall()]

            # Peak hours
            cursor = conn.execute("""
                SELECT hour_of_day, COUNT(*) as cnt
                FROM user_activity
                WHERE username = ? AND timestamp >= ?
                GROUP BY hour_of_day
                ORDER BY cnt DESC
                LIMIT 5
            """, (username, cutoff))
            peak_hours = [{"hour": row["hour_of_day"], "count": row["cnt"]} for row in cursor.fetchall()]

            conn.close()
            return {
                "total_activities_7d": total,
                "by_type": by_type,
                "top_commands": top_commands,
                "peak_hours": peak_hours,
            }
        except Exception as e:
            logger.warning("Failed to get activity stats: %s", e)
            return {"total_activities_7d": 0, "by_type": {}, "top_commands": [], "peak_hours": []}
