"""Unit/integration tests for AutopilotManager (app.autopilot.manager)."""

import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.autopilot.manager import AutopilotManager
from app.config import Config
from app.database_manager import DatabaseManager


@pytest.fixture
def project_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "sample.py").write_text("print('hi')\n", encoding="utf-8")
    return d


@pytest.fixture
def config():
    env = {"LLM_API_KEY": "test_key", "LLM_PROVIDER": "groq"}
    with patch.dict(os.environ, env, clear=True), patch("app.config.load_dotenv"):
        return Config()


@pytest.fixture
def db_manager(tmp_path):
    dbm = DatabaseManager(db_path=str(tmp_path / "test.db"))
    dbm.initialize()
    return dbm


@pytest.fixture
def manager(config, db_manager, project_dir):
    return AutopilotManager(config=config, db_manager=db_manager, project_dir=str(project_dir))


class TestControls:
    def test_disabled_by_default(self, manager):
        assert manager.status_dict()["enabled"] is False

    def test_enable_and_disable(self, manager):
        manager.enable()
        assert manager.status_dict()["enabled"] is True
        manager.disable()
        assert manager.status_dict()["enabled"] is False

    def test_record_activity_updates_timestamp(self, manager):
        before = manager.last_activity
        manager.record_activity()
        assert manager.last_activity >= before


class TestResourceGuard:
    """Phase 6: RAM/disk guard, so a Pi Zero 2W never gets pushed into OOM territory."""

    def _manager_with_monitor(self, config, db_manager, project_dir, ram_percent, disk_percent, cpu_temp_celsius=-1.0):
        monitor = MagicMock()
        monitor.get_metrics.return_value = SimpleNamespace(
            ram_percent=ram_percent, disk_percent=disk_percent, cpu_temp_celsius=cpu_temp_celsius,
        )
        mgr = AutopilotManager(
            config=config, db_manager=db_manager, project_dir=str(project_dir), system_monitor=monitor,
        )
        return mgr, monitor

    def test_blocks_when_ram_too_high(self, config, db_manager, project_dir):
        mgr, _ = self._manager_with_monitor(config, db_manager, project_dir, ram_percent=95.0, disk_percent=10.0)
        mgr.enable()
        result = mgr.run_cycle_if_due(force=True)
        assert result["reason"] == "insufficient_resources"

    def test_blocks_when_disk_too_high(self, config, db_manager, project_dir):
        mgr, _ = self._manager_with_monitor(config, db_manager, project_dir, ram_percent=10.0, disk_percent=99.0)
        result = mgr.run_cycle_if_due(force=True)
        assert result["reason"] == "insufficient_resources"

    def test_allows_when_metrics_are_fine(self, config, db_manager, project_dir):
        mgr, _ = self._manager_with_monitor(config, db_manager, project_dir, ram_percent=20.0, disk_percent=30.0)
        result = mgr.run_cycle_if_due(force=True)
        assert result["reason"] == "no_tasks"  # got past the resource guard, just nothing queued

    def test_fails_open_when_metrics_unreadable(self, config, db_manager, project_dir):
        """A -1 sentinel (e.g. /proc unavailable) means 'unknown', not '0% used' — must not block."""
        mgr, _ = self._manager_with_monitor(config, db_manager, project_dir, ram_percent=-1.0, disk_percent=-1.0)
        result = mgr.run_cycle_if_due(force=True)
        assert result["reason"] == "no_tasks"

    def test_fails_open_when_system_monitor_raises(self, config, db_manager, project_dir):
        monitor = MagicMock()
        monitor.get_metrics.side_effect = RuntimeError("boom")
        mgr = AutopilotManager(config=config, db_manager=db_manager, project_dir=str(project_dir), system_monitor=monitor)
        result = mgr.run_cycle_if_due(force=True)
        assert result["reason"] == "no_tasks"

    def test_guard_applies_even_with_force_true(self, config, db_manager, project_dir):
        """Resource limits are a hardware safety floor, not a scheduling preference force can skip."""
        mgr, _ = self._manager_with_monitor(config, db_manager, project_dir, ram_percent=99.0, disk_percent=10.0)
        mgr.task_queue.add_task("say hi", source="user")
        result = mgr.run_cycle_if_due(force=True)
        assert result == {"ran": False, "reason": "insufficient_resources", "detail": result["detail"]}
        assert "RAM" in result["detail"]

    def test_blocks_when_cpu_too_hot(self, config, db_manager, project_dir):
        mgr, _ = self._manager_with_monitor(
            config, db_manager, project_dir, ram_percent=10.0, disk_percent=10.0, cpu_temp_celsius=90.0,
        )
        result = mgr.run_cycle_if_due(force=True)
        assert result["reason"] == "insufficient_resources"
        assert "temperature" in result["detail"]

    def test_no_system_monitor_configured_never_blocks(self, manager):
        assert manager._insufficient_resources_reason() is None


class TestTimeBudget:
    def test_stuck_task_is_rolled_back_after_time_budget(self, manager, project_dir):
        manager.max_task_seconds = 0  # expires effectively immediately
        task_id = manager.task_queue.add_task("say hi", source="user")

        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "too slow"}'):
            result = manager.run_cycle_if_due(force=True)

        assert result["status"] == "rolled_back"
        assert manager.task_queue.get(task_id)["status"] == "rolled_back"



class TestRunCycleGating:
    def test_disabled_is_not_run(self, manager):
        """Disabled (coding autopilot off) is only checked once we're already in the
        quiet-hours window with no recent activity — pin the time so the result is
        deterministic regardless of when the test actually runs."""
        now = datetime(2026, 8, 24, 3, 0)
        manager.last_activity = now - timedelta(hours=2)
        with patch("app.autopilot.manager.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = manager.run_cycle_if_due()
        assert result == {"ran": False, "reason": "disabled"}

    def test_outside_window_is_not_run(self, manager):
        manager.enable()
        with patch("app.autopilot.manager.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 8, 24, 14, 0)  # 2pm, outside 02:00-06:00
            result = manager.run_cycle_if_due()
        assert result == {"ran": False, "reason": "outside_window"}

    def test_recent_chat_activity_blocks_run(self, manager):
        manager.enable()
        now = datetime(2026, 8, 24, 3, 0)  # inside the 02:00-06:00 window
        manager.last_activity = now - timedelta(minutes=5)
        with patch("app.autopilot.manager.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = manager.run_cycle_if_due()
        assert result == {"ran": False, "reason": "recent_chat_activity"}

    def test_no_tasks_available(self, manager):
        manager.enable()
        now = datetime(2026, 8, 24, 3, 0)
        manager.last_activity = now - timedelta(hours=2)
        with patch("app.autopilot.manager.datetime") as mock_dt, \
             patch("app.autopilot.manager.discover_tasks", return_value=[]):
            mock_dt.now.return_value = now
            result = manager.run_cycle_if_due()
        assert result == {"ran": False, "reason": "no_tasks"}

    def test_force_bypasses_all_gating(self, manager):
        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "ok"}'), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            manager.task_queue.add_task("say hi", source="user")
            result = manager.run_cycle_if_due(force=True)
        assert result["ran"] is True


class TestMemoryConsolidationIntegration:
    """Memory consolidation is independent of the `enabled` (coding-autopilot) toggle."""

    def test_runs_even_when_coding_autopilot_disabled(self, manager):
        mock_consolidator = MagicMock()
        mock_consolidator.is_due.return_value = True
        mock_consolidator.consolidate.return_value = {"users_processed": ["alice"], "documents_created": 1}
        manager.memory_consolidator = mock_consolidator

        now = datetime(2026, 8, 24, 3, 0)
        manager.last_activity = now - timedelta(hours=2)
        with patch("app.autopilot.manager.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = manager.run_cycle_if_due()  # note: never called manager.enable()

        assert result["ran"] is True
        assert result["action"] == "memory_consolidation"
        assert result["documents_created"] == 1
        mock_consolidator.consolidate.assert_called_once_with(now)

    def test_not_due_falls_through_to_coding_task_gate(self, manager):
        mock_consolidator = MagicMock()
        mock_consolidator.is_due.return_value = False
        manager.memory_consolidator = mock_consolidator

        now = datetime(2026, 8, 24, 3, 0)
        manager.last_activity = now - timedelta(hours=2)
        with patch("app.autopilot.manager.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = manager.run_cycle_if_due()  # coding autopilot disabled by default

        mock_consolidator.consolidate.assert_not_called()
        assert result == {"ran": False, "reason": "disabled"}

    def test_still_gated_by_resource_guard(self, config, db_manager, project_dir):
        monitor = MagicMock()
        monitor.get_metrics.return_value = SimpleNamespace(ram_percent=99.0, disk_percent=10.0, cpu_temp_celsius=-1.0)
        mgr = AutopilotManager(config=config, db_manager=db_manager, project_dir=str(project_dir), system_monitor=monitor)
        mock_consolidator = MagicMock()
        mock_consolidator.is_due.return_value = True
        mgr.memory_consolidator = mock_consolidator

        result = mgr.run_cycle_if_due(force=True)

        assert result["reason"] == "insufficient_resources"
        mock_consolidator.consolidate.assert_not_called()

    def test_status_dict_reports_pending_flag(self, manager):
        mock_consolidator = MagicMock()
        mock_consolidator.is_due.return_value = True
        manager.memory_consolidator = mock_consolidator

        assert manager.status_dict()["memory_consolidation_pending"] is True

    def test_status_dict_reports_none_when_no_consolidator(self, manager):
        assert manager.status_dict()["memory_consolidation_pending"] is None


class TestRunTaskOutcomes:
    def test_successful_task_marks_done(self, manager):
        task_id = manager.task_queue.add_task("say hi", source="user")
        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "said hi"}'), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            result = manager.run_cycle_if_due(force=True)

        assert result == {"ran": True, "task_id": task_id, "status": "done"}
        assert manager.task_queue.get(task_id)["status"] == "done"

    # Verification-failure rollback is covered end-to-end (real gate, real
    # restore) by TestRealVerificationGateIntegration.test_real_gate_fail_restores_snapshot.

    def test_ask_user_parks_as_needs_clarification_and_rolls_back(self, manager, project_dir):
        original = (project_dir / "sample.py").read_text(encoding="utf-8")
        task_id = manager.task_queue.add_task("ambiguous task", source="user")

        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "sample.py", "content": "changed"}}',
            '{"action": "ask_user", "question": "which approach do you want?"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions):
            result = manager.run_cycle_if_due(force=True)

        assert result == {"ran": True, "task_id": task_id, "status": "needs_clarification"}
        task = manager.task_queue.get(task_id)
        assert task["status"] == "needs_clarification"
        assert "which approach" in task["notes"]
        assert (project_dir / "sample.py").read_text(encoding="utf-8") == original

    def test_max_iterations_reached_rolls_back(self, manager, project_dir):
        task_id = manager.task_queue.add_task("infinite task", source="user")
        responses = ['{"action": "tool", "tool": "read_file", "args": {"path": "sample.py"}}'] * 20

        with patch("app.llm_client.GroqClient.chat", side_effect=responses):
            result = manager.run_cycle_if_due(force=True)

        assert result["status"] == "rolled_back"
        assert manager.task_queue.get(task_id)["status"] == "rolled_back"

    def test_invalid_provider_config_marks_skipped(self, manager, config):
        config.llm_provider = "carrier-pigeon"
        task_id = manager.task_queue.add_task("say hi", source="user")

        result = manager.run_cycle_if_due(force=True)

        assert result["status"] == "skipped"
        assert manager.task_queue.get(task_id)["status"] == "skipped"


class TestUiConfirmationFlow:
    def test_ui_touching_task_requires_confirmation(self, manager, project_dir):
        (project_dir / "templates").mkdir()
        (project_dir / "templates" / "page.html").write_text("<p>old</p>", encoding="utf-8")
        task_id = manager.task_queue.add_task("tweak the page layout", source="user")

        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "templates/page.html", "content": "<p>new</p>"}}',
            '{"action": "done", "summary": "updated layout"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            result = manager.run_cycle_if_due(force=True)

        assert result["status"] == "awaiting_user_confirmation"
        assert manager.task_queue.get(task_id)["status"] == "awaiting_user_confirmation"

    def test_confirm_accepted_marks_confirmed_done(self, manager, project_dir):
        (project_dir / "templates").mkdir()
        (project_dir / "templates" / "page.html").write_text("<p>old</p>", encoding="utf-8")
        task_id = manager.task_queue.add_task("tweak the ui", source="user")

        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "templates/page.html", "content": "<p>new</p>"}}',
            '{"action": "done", "summary": "updated"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            manager.run_cycle_if_due(force=True)

        result = manager.confirm_task(task_id, accepted=True)

        assert result == {"task_id": task_id, "status": "confirmed_done"}
        assert manager.task_queue.get(task_id)["status"] == "confirmed_done"
        assert (project_dir / "templates" / "page.html").read_text(encoding="utf-8") == "<p>new</p>"

    def test_confirm_rejected_restores_snapshot(self, manager, project_dir):
        (project_dir / "templates").mkdir()
        (project_dir / "templates" / "page.html").write_text("<p>old</p>", encoding="utf-8")
        task_id = manager.task_queue.add_task("tweak the ui", source="user")

        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "templates/page.html", "content": "<p>new</p>"}}',
            '{"action": "done", "summary": "updated"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            manager.run_cycle_if_due(force=True)

        result = manager.confirm_task(task_id, accepted=False)

        assert result == {"task_id": task_id, "status": "rolled_back"}
        assert (project_dir / "templates" / "page.html").read_text(encoding="utf-8") == "<p>old</p>"

    def test_confirm_unknown_task_returns_error(self, manager):
        result = manager.confirm_task(9999, accepted=True)
        assert "error" in result

    def test_confirm_task_not_awaiting_confirmation_returns_error(self, manager):
        task_id = manager.task_queue.add_task("say hi", source="user")
        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "ok"}'), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            manager.run_cycle_if_due(force=True)

        result = manager.confirm_task(task_id, accepted=True)
        assert "error" in result


class TestRealVerificationGateIntegration:
    """One end-to-end test using the real apply_verification_gate (no mocking of it),
    with a trivial always-pass/always-fail command instead of the real pytest suite."""

    def test_real_gate_pass_keeps_change(self, manager, project_dir):
        task_id = manager.task_queue.add_task(
            "say hi", source="user", verify_command='python -c "pass"'
        )
        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "ok"}'):
            result = manager.run_cycle_if_due(force=True)

        assert result["status"] == "done"
        assert manager.task_queue.get(task_id)["status"] == "done"

    def test_real_gate_fail_restores_snapshot(self, manager, project_dir):
        original = (project_dir / "sample.py").read_text(encoding="utf-8")
        task_id = manager.task_queue.add_task(
            "change sample.py", source="user", verify_command='python -c "import sys; sys.exit(1)"'
        )
        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "sample.py", "content": "changed"}}',
            '{"action": "done", "summary": "changed it"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions):
            result = manager.run_cycle_if_due(force=True)

        assert result["status"] == "rolled_back"
        assert (project_dir / "sample.py").read_text(encoding="utf-8") == original
