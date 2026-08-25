"""Tests for the autopilot_control chat tool and activity tracking (app.conversation_manager)."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from app.config import Config
from app.database_manager import DatabaseManager
from app.conversation_manager import ConversationManager


@pytest.fixture
def manager():
    os.environ["LLM_API_KEY"] = "test-key"
    os.environ["LLM_PROVIDER"] = "groq"
    config = Config()

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    db.initialize()

    mock_llm = MagicMock()
    mock_llm.chat.return_value = "Very good, Sir."
    mgr = ConversationManager(llm_client=mock_llm, config=config, db_manager=db)

    yield mgr

    db.close()
    os.unlink(path)


class TestAutopilotControlTool:
    def test_unavailable_when_no_manager_injected(self, manager):
        assert manager.autopilot_manager is None
        result = manager._tool_autopilot_control({"action": "start"})
        assert "not available" in result

    def test_start_enables_autopilot(self, manager):
        manager.autopilot_manager = MagicMock()
        manager._tool_autopilot_control({"action": "start"})
        manager.autopilot_manager.enable.assert_called_once()

    def test_pause_disables_autopilot(self, manager):
        manager.autopilot_manager = MagicMock()
        manager._tool_autopilot_control({"action": "pause"})
        manager.autopilot_manager.disable.assert_called_once()

    def test_stop_disables_autopilot(self, manager):
        manager.autopilot_manager = MagicMock()
        manager._tool_autopilot_control({"action": "stop"})
        manager.autopilot_manager.disable.assert_called_once()

    def test_status_reports_manager_state(self, manager):
        manager.autopilot_manager = MagicMock()
        manager.autopilot_manager.status_dict.return_value = {
            "enabled": True, "window": "02:00-06:00", "in_window_now": False,
            "queued_tasks": 2, "awaiting_confirmation": 1,
        }
        result = manager._tool_autopilot_control({"action": "status"})
        assert "enabled" in result
        assert "2" in result

    def test_unknown_action_reports_error(self, manager):
        manager.autopilot_manager = MagicMock()
        result = manager._tool_autopilot_control({"action": "fly"})
        assert "Unknown autopilot action" in result


class TestActivityTracking:
    def test_handle_message_records_activity(self, manager):
        manager.autopilot_manager = MagicMock()
        manager.handle_message("hello", "session-1")
        manager.autopilot_manager.record_activity.assert_called_once()
