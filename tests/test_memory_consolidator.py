"""Unit tests for MemoryConsolidator (app.memory_consolidator)."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.database_manager import DatabaseManager
from app.memory_consolidator import MemoryConsolidator, NOTHING_TO_REMEMBER


@pytest.fixture
def db_manager(tmp_path):
    dbm = DatabaseManager(db_path=str(tmp_path / "test.db"))
    dbm.initialize()
    return dbm


@pytest.fixture
def kb_manager():
    return MagicMock()


@pytest.fixture
def llm_client():
    client = MagicMock()
    client.chat.return_value = "- Prefers concise answers\n- Frequently asks about the weather in Paris"
    return client


@pytest.fixture
def consolidator(db_manager, kb_manager, llm_client):
    return MemoryConsolidator(db_manager, kb_manager, llm_client)


class TestIsDue:
    def test_due_when_never_run(self, consolidator):
        assert consolidator.is_due() is True

    def test_not_due_when_already_run_today(self, consolidator):
        now = datetime(2026, 8, 24, 3, 0)
        consolidator._set_last_consolidated_at(now)
        assert consolidator.is_due(now) is False

    def test_due_again_the_next_day(self, consolidator):
        consolidator._set_last_consolidated_at(datetime(2026, 8, 23, 3, 0))
        assert consolidator.is_due(datetime(2026, 8, 24, 3, 0)) is True


class TestConsolidate:
    def test_summarizes_and_stores_per_user_document(self, db_manager, kb_manager, llm_client):
        db_manager.save_message("s1", "user", "What's the weather in Paris?", username="alice")
        db_manager.save_message("s1", "assistant", "It's sunny, Ma'am.", username="alice")

        consolidator = MemoryConsolidator(db_manager, kb_manager, llm_client)
        result = consolidator.consolidate(datetime(2026, 8, 24, 3, 0))

        assert result["users_processed"] == ["alice"]
        assert result["documents_created"] == 1
        kb_manager.add_document.assert_called_once()
        call_args = kb_manager.add_document.call_args
        assert call_args[0][0] == "alice"
        assert "Daily memory" in call_args[0][1]
        assert call_args[1]["source"] == "auto"

    def test_skips_anonymous_sessions(self, db_manager, kb_manager, llm_client):
        db_manager.save_message("s1", "user", "hello", username=None)
        db_manager.save_message("s1", "assistant", "hi", username=None)

        consolidator = MemoryConsolidator(db_manager, kb_manager, llm_client)
        result = consolidator.consolidate(datetime(2026, 8, 24, 3, 0))

        assert result["users_processed"] == []
        assert result["documents_created"] == 0
        kb_manager.add_document.assert_not_called()

    def test_nothing_to_remember_creates_no_document(self, db_manager, kb_manager):
        db_manager.save_message("s1", "user", "what time is it?", username="bob")
        db_manager.save_message("s1", "assistant", "3pm", username="bob")
        llm_client = MagicMock()
        llm_client.chat.return_value = NOTHING_TO_REMEMBER

        consolidator = MemoryConsolidator(db_manager, kb_manager, llm_client)
        result = consolidator.consolidate(datetime(2026, 8, 24, 3, 0))

        assert result["users_processed"] == ["bob"]
        assert result["documents_created"] == 0
        kb_manager.add_document.assert_not_called()

    def test_multiple_users_processed_independently(self, db_manager, kb_manager, llm_client):
        db_manager.save_message("s1", "user", "hi", username="alice")
        db_manager.save_message("s2", "user", "hello", username="bob")

        consolidator = MemoryConsolidator(db_manager, kb_manager, llm_client)
        result = consolidator.consolidate(datetime(2026, 8, 24, 3, 0))

        assert set(result["users_processed"]) == {"alice", "bob"}
        assert kb_manager.add_document.call_count == 2

    def test_updates_last_consolidated_at(self, consolidator):
        now = datetime(2026, 8, 24, 3, 0)
        assert consolidator.get_last_consolidated_at() is None
        consolidator.consolidate(now)
        assert consolidator.get_last_consolidated_at() == now

    def test_llm_failure_is_handled_gracefully(self, db_manager, kb_manager):
        db_manager.save_message("s1", "user", "hi", username="alice")
        llm_client = MagicMock()
        llm_client.chat.side_effect = RuntimeError("provider down")

        consolidator = MemoryConsolidator(db_manager, kb_manager, llm_client)
        result = consolidator.consolidate(datetime(2026, 8, 24, 3, 0))

        assert result["documents_created"] == 0
        kb_manager.add_document.assert_not_called()

    def test_second_run_only_processes_messages_since_first_run(self, db_manager, kb_manager, llm_client):
        db_manager.save_message("s1", "user", "day 1 message", username="alice")
        consolidator = MemoryConsolidator(db_manager, kb_manager, llm_client)
        # Use real "now" so the message's real save timestamp lines up with the cutoff.
        consolidator.consolidate(datetime.now())

        result = consolidator.consolidate(datetime.now())
        assert result["users_processed"] == []
