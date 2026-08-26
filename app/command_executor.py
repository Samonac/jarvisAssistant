"""Command Executor for Jarvis Assistant.

Executes system commands with safety checks.
Auto-detects the OS and uses the appropriate shell:
- Linux/macOS: /bin/bash
- Windows: cmd.exe (or PowerShell)

Includes blocklist checking and timeout enforcement.
"""

import platform
import subprocess
from pathlib import Path
from typing import Optional


# Detect OS once at import time
IS_WINDOWS = platform.system() == "Windows"


def command_for_file(path: str) -> Optional[str]:
    """Build a platform-neutral command for a common executable file type."""
    if not path:
        return None

    quoted_path = '"' + path.replace('"', '\\"') + '"'
    suffix = Path(path).suffix.lower()
    launchers = {
        ".py": "python",
        ".js": "node",
        ".mjs": "node",
        ".cjs": "node",
        ".ts": "npx tsx",
        ".sh": "bash",
        ".bash": "bash",
        ".rb": "ruby",
        ".pl": "perl",
    }
    if suffix in launchers:
        return f"{launchers[suffix]} {quoted_path}"
    if suffix == ".ps1":
        return f"powershell -NoProfile -ExecutionPolicy Bypass -File {quoted_path}"
    if suffix in {".bat", ".cmd", ".exe", ".com"}:
        return quoted_path
    return None


class CommandExecutor:
    """Executes shell commands with safety blocklist and timeout.

    Automatically uses the correct shell for the current OS:
    - Linux/macOS: /bin/bash
    - Windows: cmd.exe

    Args:
        blocklist: List of dangerous command patterns to block.
        timeout: Maximum execution time in seconds before killing the process.
        cwd: Working directory for command execution (defaults to project dir).
    """

    def __init__(self, blocklist: list[str], timeout: int, cwd: str = None):
        self.blocklist = blocklist
        self.timeout = timeout
        self.cwd = cwd

    def execute(self, command: str) -> dict:
        """Execute a shell command and return the result.

        Checks the command against the blocklist before execution.
        If the command is blocked, returns immediately with blocked=True.
        If the command exceeds the timeout, kills the process and returns timed_out=True.

        Args:
            command: The command string to execute.

        Returns:
            A dict with keys:
                - stdout: str - captured standard output
                - stderr: str - captured standard error
                - return_code: int - process exit code (or -1 on timeout/block)
                - timed_out: bool - True if the command exceeded the timeout
                - blocked: bool - True if the command matched a blocklist pattern
                - blocked_reason: Optional[str] - the matched pattern if blocked
        """
        # Check blocklist BEFORE execution
        blocked_pattern = self._find_blocked_pattern(command)
        if blocked_pattern is not None:
            return {
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "timed_out": False,
                "blocked": True,
                "blocked_reason": f"Command matches blocked pattern: '{blocked_pattern}'",
            }

        # Replace bare "python" with the actual Python executable path
        # This avoids Windows Microsoft Store redirect issues
        import sys
        import os as _os
        if command.startswith("python ") or command.startswith("python3 "):
            prefix = "python3 " if command.startswith("python3 ") else "python "
            script_part = command[len(prefix):]

            # If the script file doesn't exist in CWD, check common subdirectories
            if self.cwd and not _os.path.isfile(_os.path.join(self.cwd, script_part)):
                for subdir in ["scripts", "plugins"]:
                    candidate = _os.path.join(self.cwd, subdir, script_part)
                    if _os.path.isfile(candidate):
                        script_part = _os.path.join(subdir, script_part)
                        break

            command = f'"{sys.executable}" {script_part}'

        # Execute the command with the appropriate shell for the OS
        try:
            if IS_WINDOWS:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=self.cwd,
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.cwd,
                )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "timed_out": False,
                "blocked": False,
                "blocked_reason": None,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "timed_out": True,
                "blocked": False,
                "blocked_reason": None,
            }

    def is_blocked(self, command: str) -> bool:
        """Check if a command matches any blocklist pattern.

        Uses case-insensitive substring matching against all patterns
        in the configured blocklist.

        Args:
            command: The command string to check.

        Returns:
            True if the command matches any blocklist pattern, False otherwise.
        """
        return self._find_blocked_pattern(command) is not None

    def _find_blocked_pattern(self, command: str) -> Optional[str]:
        """Find the first blocklist pattern that matches the command.

        Args:
            command: The command string to check.

        Returns:
            The matched pattern string, or None if no match.
        """
        command_lower = command.lower()
        for pattern in self.blocklist:
            if pattern.lower() in command_lower:
                return pattern
        return None
