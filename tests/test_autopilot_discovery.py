"""Unit tests for discover_tasks (app.autopilot.discovery)."""

from unittest.mock import MagicMock

from app.autopilot.discovery import discover_tasks


class TestDiscoverFailingTests:
    def test_parses_failed_lines_into_tasks(self, tmp_path):
        executor = MagicMock()
        executor.execute.return_value = {
            "blocked": False,
            "timed_out": False,
            "stdout": "FAILED tests/test_foo.py::test_bar - AssertionError\n1 failed in 0.1s\n",
        }

        tasks = discover_tasks(str(tmp_path), executor)

        assert any("tests/test_foo.py::test_bar" in t for t in tasks)

    def test_blocked_command_yields_no_tasks_from_tests(self, tmp_path):
        executor = MagicMock()
        executor.execute.return_value = {"blocked": True, "timed_out": False, "stdout": ""}

        tasks = discover_tasks(str(tmp_path), executor)

        assert tasks == []

    def test_timed_out_command_yields_no_tasks_from_tests(self, tmp_path):
        executor = MagicMock()
        executor.execute.return_value = {"blocked": False, "timed_out": True, "stdout": ""}

        tasks = discover_tasks(str(tmp_path), executor)

        assert tasks == []


class TestDiscoverTodos:
    def test_finds_todo_comment_in_source(self, tmp_path):
        (tmp_path / "sample.py").write_text("x = 1\n# TODO: fix this thing\ny = 2\n", encoding="utf-8")
        executor = MagicMock()
        executor.execute.return_value = {"blocked": False, "timed_out": False, "stdout": ""}

        tasks = discover_tasks(str(tmp_path), executor)

        assert any("sample.py:2" in t and "fix this thing" in t for t in tasks)

    def test_ignores_excluded_directories(self, tmp_path):
        excluded = tmp_path / "__pycache__"
        excluded.mkdir()
        (excluded / "ignored.py").write_text("# TODO: should not appear\n", encoding="utf-8")
        executor = MagicMock()
        executor.execute.return_value = {"blocked": False, "timed_out": False, "stdout": ""}

        tasks = discover_tasks(str(tmp_path), executor)

        assert tasks == []

    def test_no_todos_and_no_failures_yields_empty_list(self, tmp_path):
        (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
        executor = MagicMock()
        executor.execute.return_value = {"blocked": False, "timed_out": False, "stdout": "1 passed"}

        tasks = discover_tasks(str(tmp_path), executor)

        assert tasks == []

    def test_caps_at_max_discovered_tasks(self, tmp_path):
        lines = "\n".join(f"# TODO: item {i}" for i in range(10))
        (tmp_path / "many.py").write_text(lines + "\n", encoding="utf-8")
        executor = MagicMock()
        executor.execute.return_value = {"blocked": False, "timed_out": False, "stdout": ""}

        tasks = discover_tasks(str(tmp_path), executor)

        assert len(tasks) <= 5
