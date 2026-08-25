"""Tool wrappers for the coding agent (Phase 2).

Thin, uniform wrappers around app.file_manager.FileManager and
app.command_executor.CommandExecutor, adding a pre-write/pre-delete backup
of the previous file content so a single edit can be reverted without git.
"""

import logging
import os
import time
from typing import Optional

from app.command_executor import CommandExecutor
from app.file_manager import FileManager

logger = logging.getLogger(__name__)


class CodingAgentTools:
    """Uniform tool registry used by the coding-agent loop.

    Attributes:
        file_manager: Sandboxed file I/O.
        command_executor: Sandboxed shell execution.
        backup_dir: Directory where pre-write/pre-delete backups are written.
    """

    TOOL_NAMES = {"read_file", "write_file", "list_files", "delete_file", "run_command"}

    def __init__(self, file_manager: FileManager, command_executor: CommandExecutor, backup_dir: str):
        self.file_manager = file_manager
        self.command_executor = command_executor
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)

    def dispatch(self, tool_name: str, args: dict) -> dict:
        """Run a single tool call and return a JSON-serializable result dict."""
        if tool_name == "read_file":
            return self.file_manager.read_file(args.get("path", ""))
        if tool_name == "list_files":
            return self.file_manager.list_directory(args.get("path", "."))
        if tool_name == "write_file":
            return self._write_file_with_backup(args.get("path", ""), args.get("content", ""))
        if tool_name == "delete_file":
            return self._delete_file_with_backup(args.get("path", ""))
        if tool_name == "run_command":
            return self.command_executor.execute(args.get("command", ""))
        return {"error": f"Unknown tool: '{tool_name}'. Available: {', '.join(sorted(self.TOOL_NAMES))}"}

    def _write_file_with_backup(self, path: str, content: str) -> dict:
        existing = self.file_manager.read_file(path)
        if "content" in existing:
            self._save_backup(path, existing["content"])
        return self.file_manager.write_file(path, content)

    def _delete_file_with_backup(self, path: str) -> dict:
        existing = self.file_manager.read_file(path)
        if "content" in existing:
            self._save_backup(path, existing["content"])
        return self.file_manager.delete_file(path)

    def _save_backup(self, path: str, previous_content: str) -> Optional[str]:
        """Persist the pre-change content of a file so it can be restored later."""
        safe_name = path.replace("/", "_").replace("\\", "_").replace(":", "_")
        backup_path = os.path.join(self.backup_dir, f"{int(time.time() * 1000)}_{safe_name}.bak")
        try:
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(previous_content)
            logger.info("Backed up pre-change content of '%s' to '%s'", path, backup_path)
            return backup_path
        except OSError as e:
            logger.warning("Failed to back up '%s': %s", path, e)
            return None


def build_default_tools(config, project_dir: str) -> CodingAgentTools:
    """Build a fresh, project-sandboxed CodingAgentTools instance.

    Shared factory used by both the interactive /api/agent/* routes and the
    autopilot manager, so both modes get identical sandboxing/backup behavior.
    """
    file_manager = FileManager(base_dir=project_dir, extra_dirs=[project_dir])
    command_executor = CommandExecutor(
        blocklist=config.command_blocklist,
        timeout=config.command_timeout,
        cwd=project_dir,
    )
    backup_dir = os.path.join(project_dir, "backups", "coding_agent")
    return CodingAgentTools(file_manager, command_executor, backup_dir)
