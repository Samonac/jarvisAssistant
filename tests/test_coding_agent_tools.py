"""Unit tests for CodingAgentTools (app.coding_agent.tools)."""

import os
from unittest.mock import MagicMock

import pytest

from app.coding_agent.tools import CodingAgentTools
from app.file_manager import FileManager


@pytest.fixture
def tools(tmp_path):
    file_manager = FileManager(base_dir=str(tmp_path), extra_dirs=[str(tmp_path)])
    command_executor = MagicMock()
    backup_dir = str(tmp_path / "backups")
    return CodingAgentTools(file_manager, command_executor, backup_dir), command_executor, tmp_path


class TestDispatch:
    def test_write_then_read_round_trip(self, tools):
        agent_tools, _, _ = tools
        write_result = agent_tools.dispatch("write_file", {"path": "foo.txt", "content": "hello"})
        assert "error" not in write_result

        read_result = agent_tools.dispatch("read_file", {"path": "foo.txt"})
        assert read_result["content"] == "hello"

    def test_list_files(self, tools):
        agent_tools, _, tmp_path = tools
        (tmp_path / "a.txt").write_text("a")
        result = agent_tools.dispatch("list_files", {"path": "."})
        names = [e["name"] for e in result["entries"]]
        assert "a.txt" in names

    def test_delete_file(self, tools):
        agent_tools, _, _ = tools
        agent_tools.dispatch("write_file", {"path": "todelete.txt", "content": "bye"})
        result = agent_tools.dispatch("delete_file", {"path": "todelete.txt"})
        assert "error" not in result
        assert agent_tools.dispatch("read_file", {"path": "todelete.txt"}).get("error")

    def test_run_command_delegates_to_command_executor(self, tools):
        agent_tools, command_executor, _ = tools
        command_executor.execute.return_value = {"stdout": "ok", "stderr": "", "return_code": 0}
        result = agent_tools.dispatch("run_command", {"command": "echo hi"})
        command_executor.execute.assert_called_once_with("echo hi")
        assert result["stdout"] == "ok"

    def test_unknown_tool_returns_error(self, tools):
        agent_tools, _, _ = tools
        result = agent_tools.dispatch("teleport", {})
        assert "error" in result
        assert "teleport" in result["error"]


class TestBackupOnWriteAndDelete:
    def test_overwriting_existing_file_creates_backup(self, tools):
        agent_tools, _, _ = tools
        agent_tools.dispatch("write_file", {"path": "f.txt", "content": "version 1"})
        agent_tools.dispatch("write_file", {"path": "f.txt", "content": "version 2"})

        backups = os.listdir(agent_tools.backup_dir)
        assert len(backups) == 1
        backup_content = (agent_tools.backup_dir + os.sep + backups[0])
        with open(backup_content, encoding="utf-8") as f:
            assert f.read() == "version 1"

    def test_creating_new_file_does_not_create_backup(self, tools):
        agent_tools, _, _ = tools
        agent_tools.dispatch("write_file", {"path": "brand_new.txt", "content": "hi"})
        assert os.listdir(agent_tools.backup_dir) == []

    def test_deleting_existing_file_creates_backup(self, tools):
        agent_tools, _, _ = tools
        agent_tools.dispatch("write_file", {"path": "g.txt", "content": "keep me"})
        agent_tools.dispatch("delete_file", {"path": "g.txt"})

        backups = os.listdir(agent_tools.backup_dir)
        assert len(backups) == 1
        with open(agent_tools.backup_dir + os.sep + backups[0], encoding="utf-8") as f:
            assert f.read() == "keep me"
