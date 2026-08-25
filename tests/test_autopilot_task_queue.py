"""Unit tests for AutopilotTaskQueue (app.autopilot.task_queue)."""

import pytest

from app.autopilot.task_queue import AutopilotTaskQueue
from app.database_manager import DatabaseManager


@pytest.fixture
def queue(tmp_path):
    db_manager = DatabaseManager(db_path=str(tmp_path / "test.db"))
    db_manager.initialize()
    return AutopilotTaskQueue(db_manager)


class TestAddAndNextQueued:
    def test_next_queued_returns_none_when_empty(self, queue):
        assert queue.next_queued() is None

    def test_user_tasks_are_prioritized_over_discovered(self, queue):
        queue.add_task("discovered thing", source="discovered")
        user_id = queue.add_task("user thing", source="user")

        next_task = queue.next_queued()

        assert next_task["id"] == user_id
        assert next_task["task"] == "user thing"

    def test_multiple_user_tasks_are_fifo_after_priority(self, queue):
        first_id = queue.add_task("first", source="user")
        second_id = queue.add_task("second", source="user")

        # Newer user tasks get higher priority (pushed to the front).
        assert queue.next_queued()["id"] == second_id

    def test_discovered_tasks_come_out_in_creation_order(self, queue):
        first_id = queue.add_task("d1", source="discovered")
        queue.add_task("d2", source="discovered")

        assert queue.next_queued()["id"] == first_id


class TestUpdateStatusAndGet:
    def test_update_status_changes_status(self, queue):
        task_id = queue.add_task("do something", source="user")
        queue.update_status(task_id, "done", session_id="abc", notes="all good")

        task = queue.get(task_id)
        assert task["status"] == "done"
        assert task["session_id"] == "abc"
        assert task["notes"] == "all good"

    def test_update_status_without_session_id_preserves_existing(self, queue):
        task_id = queue.add_task("do something", source="user")
        queue.update_status(task_id, "in_progress", session_id="s1")
        queue.update_status(task_id, "done")  # no session_id passed this time

        task = queue.get(task_id)
        assert task["status"] == "done"
        assert task["session_id"] == "s1"

    def test_get_missing_task_returns_none(self, queue):
        assert queue.get(999) is None

    def test_completed_task_no_longer_returned_by_next_queued(self, queue):
        task_id = queue.add_task("do something", source="user")
        queue.update_status(task_id, "in_progress")

        assert queue.next_queued() is None


class TestListTasks:
    def test_list_tasks_filters_by_status(self, queue):
        done_id = queue.add_task("done one", source="user")
        queue.update_status(done_id, "done")
        queue.add_task("still queued", source="user")

        done_tasks = queue.list_tasks(status="done")
        assert len(done_tasks) == 1
        assert done_tasks[0]["id"] == done_id

    def test_list_tasks_without_filter_returns_all(self, queue):
        queue.add_task("a", source="user")
        queue.add_task("b", source="discovered")

        assert len(queue.list_tasks()) == 2
