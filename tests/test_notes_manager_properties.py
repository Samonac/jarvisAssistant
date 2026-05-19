"""Property tests for Notes Manager.

Tests Properties 14, 15, and 19.
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.database_manager import DatabaseManager
from app.notes_manager import NotesManager


@pytest.fixture
def notes_mgr():
    """Create a notes manager with a temporary database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    db.initialize()
    mgr = NotesManager(db, reminder_window_minutes=15)
    yield mgr
    db.close()
    os.unlink(path)


# Feature: jarvis-assistant, Property 14: Note creation and retrieval round-trip
class TestProperty14:
    """For any valid note content, saving and retrieving should include the note."""

    @given(
        content=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
        category=st.one_of(st.none(), st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L",)))),
    )
    @settings(max_examples=100)
    def test_note_creation_round_trip(self, content, category):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        mgr = NotesManager(db, reminder_window_minutes=15)
        try:
            note_id = mgr.add_note(content, category=category)
            assert note_id > 0

            active = mgr.get_all_active()
            assert any(
                n["content"] == content
                and n["status"] == "active"
                for n in active
            )
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 15: Completing a note removes it from active list
class TestProperty15:
    """After marking a note as completed, it should not appear in active notes."""

    @given(
        content=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=100)
    def test_completing_note_removes_from_active(self, content):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        mgr = NotesManager(db, reminder_window_minutes=15)
        try:
            note_id = mgr.add_note(content)
            success = mgr.complete_note(note_id)
            assert success

            active = mgr.get_all_active()
            assert not any(n["id"] == note_id for n in active)
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 19: Note search returns relevant results
class TestProperty19:
    """For any note with a specific keyword, searching for that keyword returns it."""

    @given(
        keyword=st.text(min_size=4, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
    )
    @settings(max_examples=100)
    def test_note_search_returns_relevant(self, keyword):
        assume(keyword.strip())  # Skip empty after strip
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        mgr = NotesManager(db, reminder_window_minutes=15)
        try:
            content = f"Remember to {keyword} tomorrow"
            mgr.add_note(content)

            results = mgr.search(keyword)
            assert any(keyword in r["content"] for r in results)
        finally:
            db.close()
            os.unlink(path)
