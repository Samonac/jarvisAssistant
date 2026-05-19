"""Notes Manager for Jarvis Assistant.

High-level interface for mental notes — reminders, to-do items, and
contextual information the assistant needs to remember for the user.

Provides keyword-based relevance matching and due-date awareness
for proactive recall during conversations.
"""

import logging
from datetime import datetime
from typing import Optional

from app.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class NotesManager:
    """Manages mental notes with contextual relevance and proactive recall.

    Attributes:
        db_manager: Database manager for persistent note storage.
        reminder_window_minutes: How far ahead to look for due notes.
    """

    def __init__(self, db_manager: DatabaseManager, reminder_window_minutes: int = 15):
        self.db_manager = db_manager
        self.reminder_window_minutes = reminder_window_minutes

    def add_note(
        self,
        content: str,
        due_date: Optional[datetime] = None,
        category: Optional[str] = None,
        command: Optional[str] = None,
        recurrence: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> int:
        """Store a new mental note.

        Args:
            content: The note text.
            due_date: Optional datetime when this note is due.
            category: Optional category tag for organization.
            command: Optional command to execute when due.
            recurrence: Optional recurrence pattern ("daily", "hourly", "weekly", "every Nm", "every Nh").
            expires_at: Optional datetime after which the note auto-clears.

        Returns:
            The ID of the newly created note.
        """
        return self.db_manager.save_note(
            content, due_date=due_date, category=category,
            command=command, recurrence=recurrence, expires_at=expires_at
        )

    def get_all_active(self) -> list[dict]:
        """Get all active (non-completed) notes."""
        return self.db_manager.get_active_notes()

    def complete_note(self, note_id: int) -> bool:
        """Mark a note as done.

        Args:
            note_id: The ID of the note to complete.

        Returns:
            True if the note was found and completed, False otherwise.
        """
        return self.db_manager.complete_note(note_id)

    def search(self, query: str) -> list[dict]:
        """Search notes by content or category.

        Args:
            query: Search term to match against note content.

        Returns:
            List of matching notes.
        """
        return self.db_manager.search_notes(query)

    def get_relevant_notes(self, context: str) -> list[dict]:
        """Get notes relevant to the current conversation context.

        Uses keyword matching: extracts significant words from the context
        and searches notes for matches.

        Args:
            context: The current user message or conversation context.

        Returns:
            List of notes that are contextually relevant.
        """
        if not context or not context.strip():
            return []

        # Extract keywords (words > 3 chars, skip common stop words)
        stop_words = {
            "that", "this", "with", "from", "have", "been", "will", "what",
            "when", "where", "which", "there", "their", "about", "would",
            "could", "should", "just", "like", "your", "they", "them",
            "some", "than", "then", "also", "into", "more", "very",
            "please", "thank", "thanks", "hello", "jarvis",
        }

        words = context.lower().split()
        keywords = [
            w.strip(".,!?;:'\"()[]{}") for w in words
            if len(w) > 3 and w.lower().strip(".,!?;:'\"()[]{}") not in stop_words
        ]

        if not keywords:
            return []

        # Search for each keyword and collect unique results
        seen_ids = set()
        relevant = []
        for keyword in keywords[:5]:  # Limit to 5 keywords to avoid excessive queries
            if not keyword:
                continue
            results = self.db_manager.search_notes(keyword)
            for note in results:
                if note["id"] not in seen_ids and note["status"] == "active":
                    seen_ids.add(note["id"])
                    relevant.append(note)

        return relevant[:5]  # Limit to 5 relevant notes to conserve context

    def get_timely_notes(self) -> list[dict]:
        """Get notes with approaching due dates (within reminder window).

        Returns:
            List of notes whose due_date is within the reminder window.
        """
        return self.db_manager.get_due_notes(window_minutes=self.reminder_window_minutes)

    def format_for_context(self, notes: list[dict]) -> str:
        """Format notes as a string suitable for injection into LLM context.

        Args:
            notes: List of note dicts to format.

        Returns:
            Formatted string representation of the notes.
        """
        if not notes:
            return "No active notes."

        lines = []
        for note in notes:
            prefix = f"[#{note['id']}]"
            category_str = f" ({note['category']})" if note.get("category") else ""
            due_str = f" [Due: {note['due_date']}]" if note.get("due_date") else ""
            status_str = " ✓" if note.get("status") == "completed" else ""
            lines.append(f"{prefix}{category_str}{due_str}{status_str} {note['content']}")

        return "\n".join(lines)
