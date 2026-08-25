"""Nightly conversation memory consolidation.

Turns the day's raw Q&A conversation history into durable per-user memory
entries in the knowledge base, so Jarvis remembers past exchanges and
gradually learns each user's typical requirements — without the user having
to explicitly teach it anything. The resulting entries are automatically
picked up by the existing KnowledgeBaseManager.get_context_for_message()
retrieval already used in every conversation
(see ConversationManager._build_messages).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

NOTHING_TO_REMEMBER = "NOTHING_TO_REMEMBER"

CONSOLIDATION_PROMPT_TEMPLATE = """You are maintaining a long-term memory profile for a user of the J.A.R.V.I.S. assistant.

Below is a transcript of today's conversation(s) with this user. Extract durable,
useful memory: stable facts about the user, their preferences, recurring
requests, and anything that would help you serve them better in future
conversations. Do NOT include one-off trivia (e.g. what the time was) or
anything only relevant to that single moment.

If there is genuinely nothing worth remembering, respond with exactly: {nothing_marker}

Otherwise, respond with a concise bullet list (no more than 10 bullets) of durable memory facts.
Respond with ONLY the marker or the bullet list — no other commentary.

--- TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---
"""

# Keep the summarization prompt bounded regardless of how chatty a day was.
MAX_TRANSCRIPT_CHARS = 8000


class MemoryConsolidator:
    """Summarizes a day's conversations per user into knowledge-base memory entries."""

    def __init__(self, db_manager, kb_manager, llm_client):
        self.db_manager = db_manager
        self.kb_manager = kb_manager
        self.llm_client = llm_client
        self._init_tables()

    def _get_conn(self):
        import sqlite3
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        try:
            conn = self._get_conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_consolidation_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_consolidated_at DATETIME
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO memory_consolidation_state (id, last_consolidated_at) VALUES (1, NULL)"
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init memory_consolidation_state table: %s", e)

    def get_last_consolidated_at(self) -> Optional[datetime]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT last_consolidated_at FROM memory_consolidation_state WHERE id = 1"
        ).fetchone()
        conn.close()
        if row is None or row["last_consolidated_at"] is None:
            return None
        return datetime.fromisoformat(row["last_consolidated_at"])

    def _set_last_consolidated_at(self, when: datetime) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE memory_consolidation_state SET last_consolidated_at = ? WHERE id = 1",
            (when.isoformat(),),
        )
        conn.commit()
        conn.close()

    def is_due(self, now: Optional[datetime] = None) -> bool:
        """True if consolidation hasn't run yet today (or has never run)."""
        now = now or datetime.now()
        last = self.get_last_consolidated_at()
        return last is None or last.date() < now.date()

    def consolidate(self, now: Optional[datetime] = None) -> dict:
        """Summarize all messages since the last run, grouped by user, into KB memory entries.

        Returns {"users_processed": [...], "documents_created": N}.
        """
        now = now or datetime.now()
        since = self.get_last_consolidated_at() or (now - timedelta(days=1))

        messages_by_user = self.db_manager.get_messages_since(since)
        documents_created = 0
        users_processed = []

        for username, messages in messages_by_user.items():
            if not username:
                continue  # Anonymous/no-login sessions have no profile to attach memory to.

            transcript = self._format_transcript(messages)
            if not transcript.strip():
                continue

            summary = self._summarize(transcript)
            users_processed.append(username)

            if not summary or summary.strip().upper() == NOTHING_TO_REMEMBER:
                continue

            title = f"Daily memory — {now.strftime('%Y-%m-%d')}"
            self.kb_manager.add_document(username, title, summary.strip(), source="auto")
            documents_created += 1

        self._set_last_consolidated_at(now)
        logger.info(
            "Memory consolidation complete: %d user(s) processed, %d memory document(s) created",
            len(users_processed), documents_created,
        )
        return {"users_processed": users_processed, "documents_created": documents_created}

    @staticmethod
    def _format_transcript(messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = "User" if m["role"] == "user" else "Jarvis"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    def _summarize(self, transcript: str) -> str:
        prompt = CONSOLIDATION_PROMPT_TEMPLATE.format(
            transcript=transcript[:MAX_TRANSCRIPT_CHARS], nothing_marker=NOTHING_TO_REMEMBER,
        )
        try:
            return self.llm_client.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            logger.error("Memory consolidation summarization failed: %s", e)
            return ""
