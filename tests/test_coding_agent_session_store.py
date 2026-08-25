"""Unit tests for AgentSessionStore (app.coding_agent.session_store)."""

import pytest

from app.coding_agent.session_store import AgentSessionStore
from app.coding_agent.state import AgentTaskState, ToolCallRecord
from app.database_manager import DatabaseManager


@pytest.fixture
def store(tmp_path):
    db_manager = DatabaseManager(db_path=str(tmp_path / "test.db"))
    db_manager.initialize()
    return AgentSessionStore(db_manager)


class TestSaveAndLoad:
    def test_round_trips_all_fields(self, store):
        state = AgentTaskState(
            session_id="abc123",
            task="fix the bug",
            status="awaiting_user",
            iteration=2,
            provider="groq",
            model="custom-model",
            effort="deep",
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "fix the bug"}],
            history=[ToolCallRecord(tool="read_file", args={"path": "a.py"}, output={"content": "x"}, thought="checking")],
            pending_question="Which file?",
        )

        store.save(state)
        loaded = store.load("abc123")

        assert loaded is not None
        assert loaded.session_id == "abc123"
        assert loaded.task == "fix the bug"
        assert loaded.status == "awaiting_user"
        assert loaded.iteration == 2
        assert loaded.provider == "groq"
        assert loaded.model == "custom-model"
        assert loaded.effort == "deep"
        assert loaded.messages == state.messages
        assert loaded.pending_question == "Which file?"
        assert len(loaded.history) == 1
        assert loaded.history[0].tool == "read_file"
        assert loaded.history[0].output == {"content": "x"}

    def test_load_missing_session_returns_none(self, store):
        assert store.load("does-not-exist") is None

    def test_save_is_idempotent_upsert(self, store):
        state = AgentTaskState(session_id="s1", task="t", status="in_progress")
        store.save(state)

        state.status = "done"
        state.summary = "all done"
        store.save(state)

        loaded = store.load("s1")
        assert loaded.status == "done"
        assert loaded.summary == "all done"
