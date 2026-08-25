"""Persistent storage for coding-agent sessions (Phase 3).

Generalizes the pause/resume pattern already used in
app.conversation_manager.ConversationManager._pending_emails (an in-memory,
session_id-keyed dict checked at the top of the next request), but persists
to SQLite so an ask_user pause survives a process restart and can be resumed
with the answer folded back into the exact same transcript/plan.
"""

import json
import logging
import sqlite3
from typing import Optional

from app.coding_agent.state import AgentTaskState, ToolCallRecord

logger = logging.getLogger(__name__)


class AgentSessionStore:
    """Persists AgentTaskState rows keyed by session_id."""

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
                CREATE TABLE IF NOT EXISTS coding_agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    iteration INTEGER DEFAULT 0,
                    provider TEXT,
                    model TEXT,
                    effort TEXT,
                    messages TEXT NOT NULL,
                    history TEXT NOT NULL,
                    pending_question TEXT,
                    summary TEXT,
                    error TEXT,
                    verification TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Migration for DBs created before the `verification` column existed.
            try:
                conn.execute("ALTER TABLE coding_agent_sessions ADD COLUMN verification TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init coding_agent_sessions table: %s", e)

    def save(self, state: AgentTaskState) -> None:
        """Insert or update the persisted row for this session."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO coding_agent_sessions
                (session_id, task, status, iteration, provider, model, effort,
                 messages, history, pending_question, summary, error, verification, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                status=excluded.status,
                iteration=excluded.iteration,
                messages=excluded.messages,
                history=excluded.history,
                pending_question=excluded.pending_question,
                summary=excluded.summary,
                error=excluded.error,
                verification=excluded.verification,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                state.session_id,
                state.task,
                state.status,
                state.iteration,
                state.provider,
                state.model,
                state.effort,
                json.dumps(state.messages),
                json.dumps([_record_to_dict(h) for h in state.history]),
                state.pending_question,
                state.summary,
                state.error,
                json.dumps(state.verification) if state.verification is not None else None,
            ),
        )
        conn.commit()
        conn.close()

    def load(self, session_id: str) -> Optional[AgentTaskState]:
        """Load a persisted session by ID, or None if it doesn't exist."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM coding_agent_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None

        history = [
            ToolCallRecord(
                tool=h["tool"],
                args=h["args"],
                output=h["output"],
                thought=h.get("thought", ""),
                timestamp=h.get("timestamp", ""),
            )
            for h in json.loads(row["history"])
        ]
        return AgentTaskState(
            session_id=row["session_id"],
            task=row["task"],
            status=row["status"],
            iteration=row["iteration"],
            provider=row["provider"] or "",
            model=row["model"],
            effort=row["effort"] or "standard",
            messages=json.loads(row["messages"]),
            history=history,
            pending_question=row["pending_question"],
            summary=row["summary"],
            error=row["error"],
            verification=json.loads(row["verification"]) if row["verification"] else None,
        )


def _record_to_dict(record: ToolCallRecord) -> dict:
    return {
        "tool": record.tool,
        "args": record.args,
        "output": record.output,
        "thought": record.thought,
        "timestamp": record.timestamp,
    }
