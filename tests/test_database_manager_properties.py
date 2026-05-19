"""Property tests for Database Manager.

Tests Properties 12, 13, 18, and 26.
"""

import os
import tempfile

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.database_manager import DatabaseManager


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    manager = DatabaseManager(db_path=path)
    manager.initialize()
    yield manager
    manager.close()
    os.unlink(path)


# Feature: jarvis-assistant, Property 12: Database message persistence round-trip
class TestProperty12:
    """For any valid session_id, role, and content, saving and retrieving
    should include the saved message."""

    @given(
        session_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        role=st.sampled_from(["user", "assistant"]),
        content=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=100)
    def test_message_persistence_round_trip(self, session_id, role, content):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        try:
            db.save_message(session_id, role, content)
            history = db.get_history(session_id, max_pairs=10)
            assert any(
                m["session_id"] == session_id
                and m["role"] == role
                and m["content"] == content
                for m in history
            )
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 13: History retrieval respects pair limit
class TestProperty13:
    """For any session with more than 10 pairs, retrieval with max_pairs=10
    returns exactly 20 messages (10 pairs) and they are the most recent."""

    @given(
        num_pairs=st.integers(min_value=11, max_value=25),
    )
    @settings(max_examples=50)
    def test_history_respects_pair_limit(self, num_pairs):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        try:
            session_id = "test-session"
            for i in range(num_pairs):
                db.save_message(session_id, "user", f"msg-{i}")
                db.save_message(session_id, "assistant", f"reply-{i}")

            history = db.get_history(session_id, max_pairs=10)
            assert len(history) == 20  # 10 pairs = 20 messages

            # Verify they are the most recent
            last_user_msg = history[-2]
            assert last_user_msg["content"] == f"msg-{num_pairs - 1}"
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 18: Database initialization is idempotent
class TestProperty18:
    """Calling initialize() N times produces the same schema and preserves data."""

    @given(n_calls=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_initialization_idempotent(self, n_calls):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        try:
            # Save some data
            db.save_message("s1", "user", "hello")
            db.save_note("test note")

            # Call initialize multiple times
            for _ in range(n_calls):
                db.initialize()

            # Verify data is preserved
            history = db.get_history("s1", max_pairs=10)
            assert any(m["content"] == "hello" for m in history)

            notes = db.get_active_notes()
            assert any(n["content"] == "test note" for n in notes)
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 26: Metrics schema idempotence
class TestProperty26:
    """Calling initialize() N times preserves existing metric data."""

    @given(n_calls=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_metrics_schema_idempotent(self, n_calls):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        try:
            # Save a metric event
            db.save_metric_event("llm_call", duration_ms=100.0, success=True, session_id="s1")

            # Call initialize multiple times
            for _ in range(n_calls):
                db.initialize()

            # Verify metric data is preserved
            summary = db.get_metrics_summary()
            assert summary["total_calls"] >= 1
        finally:
            db.close()
            os.unlink(path)
