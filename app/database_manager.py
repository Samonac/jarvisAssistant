"""Database Manager for Jarvis Assistant.

Manages SQLite database for persistent conversation history, mental notes,
and usage metrics. Uses WAL mode for better concurrent read performance.
Falls back to in-memory storage on database corruption or permission errors.
"""

import logging
import sqlite3
import threading
from functools import wraps
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def _database_locked(method):
    """Serialize access to the shared SQLite connection across Flask threads."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT,
    due_date DATETIME,
    status TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    command TEXT,
    recurrence TEXT,
    expires_at DATETIME
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    tool_name TEXT,
    duration_ms REAL,
    success INTEGER DEFAULT 1,
    session_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_notes_status ON notes(status);
CREATE INDEX IF NOT EXISTS idx_notes_due_date ON notes(due_date);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_event_type ON metrics(event_type);

CREATE TABLE IF NOT EXISTS session_metadata (
    session_id TEXT PRIMARY KEY,
    title TEXT
);
"""


class DatabaseManager:
    """Manages SQLite database for persistent storage.

    Provides methods for conversation history, notes CRUD, and metrics.
    Falls back to in-memory SQLite if the database file is corrupted
    or inaccessible.
    """

    def __init__(self, db_path: str = "jarvis.db"):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._using_memory = False
        self._lock = threading.RLock()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection with WAL mode."""
        if self._connection is not None:
            return self._connection

        try:
            self._connection = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.row_factory = sqlite3.Row
            self._using_memory = False
        except (sqlite3.DatabaseError, OSError, PermissionError) as e:
            logger.error(
                "Failed to open database at '%s': %s. Falling back to in-memory storage.",
                self.db_path,
                e,
            )
            self._connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.row_factory = sqlite3.Row
            self._using_memory = True

        return self._connection

    @_database_locked
    def initialize(self) -> None:
        """Create tables if they don't exist. Called at startup.

        This method is idempotent — calling it multiple times produces
        the same schema as calling it once, and existing data is preserved.
        Also handles schema migrations for new columns.
        """
        conn = self._get_connection()
        try:
            conn.executescript(SCHEMA)
            # Migrate: add new columns if they don't exist (for existing DBs)
            self._migrate(conn)
            conn.commit()
        except sqlite3.DatabaseError as e:
            logger.error("Database error during initialization: %s", e)
            # Fall back to in-memory if schema creation fails
            self._connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.row_factory = sqlite3.Row
            self._using_memory = True
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add missing columns to existing tables (safe migrations)."""
        # Check which columns exist in the notes table
        cursor = conn.execute("PRAGMA table_info(notes)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if "command" not in existing_columns:
            conn.execute("ALTER TABLE notes ADD COLUMN command TEXT")
            logger.info("Migration: added 'command' column to notes table")

        if "recurrence" not in existing_columns:
            conn.execute("ALTER TABLE notes ADD COLUMN recurrence TEXT")
            logger.info("Migration: added 'recurrence' column to notes table")

        if "expires_at" not in existing_columns:
            conn.execute("ALTER TABLE notes ADD COLUMN expires_at DATETIME")
            logger.info("Migration: added 'expires_at' column to notes table")

        # Add username to conversations for per-user filtering
        cursor = conn.execute("PRAGMA table_info(conversations)")
        conv_columns = {row[1] for row in cursor.fetchall()}
        if "username" not in conv_columns:
            conn.execute("ALTER TABLE conversations ADD COLUMN username TEXT")
            logger.info("Migration: added 'username' column to conversations table")

        # Add username to metrics for per-user filtering
        cursor = conn.execute("PRAGMA table_info(metrics)")
        metrics_columns = {row[1] for row in cursor.fetchall()}
        if "username" not in metrics_columns:
            conn.execute("ALTER TABLE metrics ADD COLUMN username TEXT")
            logger.info("Migration: added 'username' column to metrics table")

    @_database_locked
    def save_message(self, session_id: str, role: str, content: str, username: str = None) -> None:
        """Persist a single message to the conversations table."""
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, username) VALUES (?, ?, ?, ?)",
            (session_id, role, content, username),
        )
        conn.commit()

    @_database_locked
    def get_sessions(self, limit: int = 50, username: str = None) -> list[dict]:
        """Retrieve a list of past conversation sessions.

        If username is provided, only returns sessions belonging to that user.
        If username is None (admin view), returns all sessions.
        """
        conn = self._get_connection()

        if username:
            cursor = conn.execute(
                """
                SELECT session_id,
                       MIN(timestamp) as started_at,
                       MAX(timestamp) as last_active,
                       COUNT(*) as message_count,
                       title
                FROM conversations
                LEFT JOIN (SELECT session_id as sid, title FROM session_metadata) sm
                    ON conversations.session_id = sm.sid
                WHERE conversations.username = ?
                GROUP BY session_id
                ORDER BY MAX(timestamp) DESC
                LIMIT ?
                """,
                (username, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT session_id,
                       MIN(timestamp) as started_at,
                       MAX(timestamp) as last_active,
                       COUNT(*) as message_count,
                       title,
                       username
                FROM conversations
                LEFT JOIN (SELECT session_id as sid, title FROM session_metadata) sm
                    ON conversations.session_id = sm.sid
                GROUP BY session_id
                ORDER BY MAX(timestamp) DESC
                LIMIT ?
                """,
                (limit,),
            )
        sessions = []
        for row in cursor.fetchall():
            # Get the first user message as preview
            preview_cursor = conn.execute(
                "SELECT content FROM conversations WHERE session_id = ? AND role = 'user' ORDER BY timestamp ASC LIMIT 1",
                (row["session_id"],),
            )
            preview_row = preview_cursor.fetchone()
            preview = preview_row["content"][:80] if preview_row else "Empty conversation"

            sessions.append({
                "session_id": row["session_id"],
                "started_at": row["started_at"],
                "last_active": row["last_active"],
                "message_count": row["message_count"],
                "preview": preview,
                "title": row["title"] or "",
            })
        return sessions

    @_database_locked
    def save_session_title(self, session_id: str, title: str) -> None:
        """Save or update a generated title for a session."""
        conn = self._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO session_metadata (session_id, title) VALUES (?, ?)",
            (session_id, title),
        )
        conn.commit()

    @_database_locked
    def get_full_history(self, session_id: str) -> list[dict]:
        """Retrieve the full conversation history for a session (no limit).

        Returns all messages ordered by timestamp ascending.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT session_id, role, content, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            (session_id,),
        )
        return [
            {
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            for row in cursor.fetchall()
        ]

    @_database_locked
    def search_conversations(self, query: str, limit: int = 20) -> list[dict]:
        """Search across all conversations for messages matching a query.

        Returns matching messages grouped by session, with context.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT c.session_id, c.role, c.content, c.timestamp,
                   sm.title
            FROM conversations c
            LEFT JOIN session_metadata sm ON c.session_id = sm.session_id
            WHERE c.content LIKE ?
            ORDER BY c.timestamp DESC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "title": row["title"] or "",
            })
        return results

    @_database_locked
    def get_history(self, session_id: str, max_pairs: int = 10) -> list[dict]:
        """Retrieve the most recent N message pairs for a session.

        Returns messages ordered by timestamp ascending (oldest first),
        limited to the most recent max_pairs * 2 messages.
        """
        conn = self._get_connection()
        # Get the most recent max_pairs*2 messages, then return in chronological order
        cursor = conn.execute(
            """
            SELECT session_id, role, content, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (session_id, max_pairs * 2),
        )
        rows = cursor.fetchall()
        # Reverse to get chronological order
        rows = list(reversed(rows))
        return [
            {
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    @_database_locked
    def prune_old_conversations(self, retention_days: int = 30) -> int:
        """Delete conversations older than retention period.

        Returns the count of deleted records.
        """
        conn = self._get_connection()
        cutoff = datetime.now() - timedelta(days=retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            "DELETE FROM conversations WHERE timestamp < ?",
            (cutoff_str,),
        )
        conn.commit()
        return cursor.rowcount

    @_database_locked
    def get_messages_since(self, since: datetime) -> dict[str, list[dict]]:
        """Retrieve all messages recorded after `since`, grouped by username.

        Used by the nightly memory consolidator to summarize a day's Q&A per
        user. Messages from anonymous/no-login sessions (no username) are
        grouped under the empty-string key.
        """
        conn = self._get_connection()
        since_str = since.strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            """
            SELECT username, role, content, timestamp
            FROM conversations
            WHERE timestamp > ?
            ORDER BY username, timestamp ASC, id ASC
            """,
            (since_str,),
        )
        grouped: dict[str, list[dict]] = {}
        for row in cursor.fetchall():
            username = row["username"] or ""
            grouped.setdefault(username, []).append({
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            })
        return grouped

    # --- Notes CRUD ---

    @_database_locked
    def save_note(
        self,
        content: str,
        due_date: Optional[datetime] = None,
        category: Optional[str] = None,
        command: Optional[str] = None,
        recurrence: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> int:
        """Save a mental note. Returns the note ID.
        
        Args:
            content: The note text.
            due_date: When the note is due.
            category: Optional category tag.
            command: Optional command to execute when due.
            recurrence: Optional recurrence pattern (e.g., "daily", "hourly", "weekly", "every 5m").
            expires_at: Optional expiry datetime after which the note is auto-cleared.
        """
        conn = self._get_connection()
        due_date_str = due_date.strftime("%Y-%m-%d %H:%M:%S") if due_date else None
        expires_at_str = expires_at.strftime("%Y-%m-%d %H:%M:%S") if expires_at else None
        cursor = conn.execute(
            "INSERT INTO notes (content, category, due_date, command, recurrence, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
            (content, category, due_date_str, command, recurrence, expires_at_str),
        )
        conn.commit()
        return cursor.lastrowid

    @_database_locked
    def get_active_notes(self) -> list[dict]:
        """Retrieve all notes with status='active'."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id, content, category, due_date, status, created_at, completed_at, command, recurrence "
            "FROM notes WHERE status = 'active' ORDER BY created_at DESC"
        )
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "category": row["category"],
                "due_date": row["due_date"],
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "command": row["command"] if "command" in row.keys() else None,
                "recurrence": row["recurrence"] if "recurrence" in row.keys() else None,
            }
            for row in cursor.fetchall()
        ]

    @_database_locked
    def complete_note(self, note_id: int) -> bool:
        """Mark a note as completed. Returns True if the note was found and updated."""
        conn = self._get_connection()
        completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            "UPDATE notes SET status = 'completed', completed_at = ? WHERE id = ? AND status = 'active'",
            (completed_at, note_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    @_database_locked
    def search_notes(self, query: str) -> list[dict]:
        """Search notes by content using LIKE matching."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id, content, category, due_date, status, created_at, completed_at "
            "FROM notes WHERE content LIKE ? ORDER BY created_at DESC",
            (f"%{query}%",),
        )
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "category": row["category"],
                "due_date": row["due_date"],
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }
            for row in cursor.fetchall()
        ]

    @_database_locked
    def get_due_notes(self, window_minutes: int = 15) -> list[dict]:
        """Get notes with due dates within the reminder window.

        Returns active notes whose due_date is between now and now + window_minutes.
        """
        conn = self._get_connection()
        now = datetime.now()
        window_end = now + timedelta(minutes=window_minutes)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        window_end_str = window_end.strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            "SELECT id, content, category, due_date, status, created_at, completed_at "
            "FROM notes WHERE status = 'active' AND due_date IS NOT NULL "
            "AND due_date >= ? AND due_date <= ? ORDER BY due_date ASC",
            (now_str, window_end_str),
        )
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "category": row["category"],
                "due_date": row["due_date"],
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }
            for row in cursor.fetchall()
        ]

    # --- Metrics methods ---

    @_database_locked
    def save_metric_event(
        self,
        event_type: str,
        tool_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        session_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> None:
        """Persist a metric event to the metrics table."""
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO metrics (event_type, tool_name, duration_ms, success, session_id, username) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event_type, tool_name, duration_ms, 1 if success else 0, session_id, username),
        )
        conn.commit()

    @_database_locked
    def get_metrics_summary(self) -> dict:
        """Return aggregated metrics summary.

        Returns a dict with: total_calls, calls_today, avg_response_ms,
        p95_response_ms, tool_usage, error_rate, active_sessions.
        """
        conn = self._get_connection()

        # Total calls
        row = conn.execute("SELECT COUNT(*) as cnt FROM metrics").fetchone()
        total_calls = row["cnt"]

        # Calls today
        today_str = datetime.now().strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM metrics WHERE date(timestamp) = ?",
            (today_str,),
        ).fetchone()
        calls_today = row["cnt"]

        # Average response time (from LLM calls with duration)
        row = conn.execute(
            "SELECT AVG(duration_ms) as avg_ms FROM metrics "
            "WHERE event_type = 'llm_call' AND duration_ms IS NOT NULL"
        ).fetchone()
        avg_response_ms = row["avg_ms"] if row["avg_ms"] is not None else 0.0

        # P95 response time
        p95_response_ms = self._calculate_p95()

        # Tool usage breakdown
        cursor = conn.execute(
            "SELECT tool_name, COUNT(*) as cnt FROM metrics "
            "WHERE event_type = 'tool_call' AND tool_name IS NOT NULL "
            "GROUP BY tool_name"
        )
        tool_usage = {row["tool_name"]: row["cnt"] for row in cursor.fetchall()}

        # Error rate
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM metrics WHERE success = 0"
        ).fetchone()
        error_count = row["cnt"]
        error_rate = error_count / total_calls if total_calls > 0 else 0.0

        # Active sessions (distinct session_ids in last 24 hours)
        yesterday_str = (datetime.now() - timedelta(hours=24)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) as cnt FROM metrics "
            "WHERE session_id IS NOT NULL AND timestamp >= ?",
            (yesterday_str,),
        ).fetchone()
        active_sessions = row["cnt"]

        return {
            "total_calls": total_calls,
            "calls_today": calls_today,
            "avg_response_ms": avg_response_ms,
            "p95_response_ms": p95_response_ms,
            "tool_usage": tool_usage,
            "error_rate": error_rate,
            "active_sessions": active_sessions,
        }

    def _calculate_p95(self) -> float:
        """Calculate the 95th percentile response time for LLM calls."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT duration_ms FROM metrics "
            "WHERE event_type = 'llm_call' AND duration_ms IS NOT NULL "
            "ORDER BY duration_ms ASC"
        )
        durations = [row["duration_ms"] for row in cursor.fetchall()]
        if not durations:
            return 0.0
        idx = int(len(durations) * 0.95)
        idx = min(idx, len(durations) - 1)
        return durations[idx]

    @_database_locked
    def get_daily_breakdown(self, days: int = 7) -> list[dict]:
        """Return per-day metrics for the last N days."""
        conn = self._get_connection()
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        cursor = conn.execute(
            """
            SELECT date(timestamp) as day,
                   COUNT(*) as total_calls,
                   AVG(CASE WHEN event_type = 'llm_call' THEN duration_ms END) as avg_response_ms,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
            FROM metrics
            WHERE date(timestamp) >= ?
            GROUP BY date(timestamp)
            ORDER BY day ASC
            """,
            (start_date,),
        )
        return [
            {
                "date": row["day"],
                "total_calls": row["total_calls"],
                "avg_response_ms": row["avg_response_ms"] if row["avg_response_ms"] is not None else 0.0,
                "error_count": row["error_count"],
            }
            for row in cursor.fetchall()
        ]

    @_database_locked
    def get_hourly_breakdown(self, hours: int = 24) -> list[dict]:
        """Return per-hour metrics for the last N hours.

        Returns a list of dicts with: hour, total_calls, avg_response_ms, error_count.
        """
        conn = self._get_connection()
        start_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            """
            SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour,
                   COUNT(*) as total_calls,
                   AVG(CASE WHEN event_type = 'llm_call' THEN duration_ms END) as avg_response_ms,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
            FROM metrics
            WHERE timestamp >= ?
            GROUP BY strftime('%Y-%m-%d %H:00', timestamp)
            ORDER BY hour ASC
            """,
            (start_time,),
        )
        return [
            {
                "hour": row["hour"],
                "total_calls": row["total_calls"],
                "avg_response_ms": row["avg_response_ms"] if row["avg_response_ms"] is not None else 0.0,
                "error_count": row["error_count"],
            }
            for row in cursor.fetchall()
        ]

    @_database_locked
    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
