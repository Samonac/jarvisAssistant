"""Unit tests for run_verification (app.coding_agent.verify)."""

from unittest.mock import MagicMock

from app.coding_agent.verify import DEFAULT_VERIFY_COMMAND, run_verification


class TestRunVerification:
    def test_zero_exit_code_passes(self):
        executor = MagicMock()
        executor.execute.return_value = {"return_code": 0, "blocked": False, "timed_out": False, "stdout": "ok"}

        result = run_verification(executor)

        assert result.passed is True
        assert result.command == DEFAULT_VERIFY_COMMAND
        executor.execute.assert_called_once_with(DEFAULT_VERIFY_COMMAND)

    def test_nonzero_exit_code_fails(self):
        executor = MagicMock()
        executor.execute.return_value = {"return_code": 1, "blocked": False, "timed_out": False, "stderr": "boom"}

        result = run_verification(executor)

        assert result.passed is False

    def test_timed_out_fails_even_with_zero_return_code(self):
        executor = MagicMock()
        executor.execute.return_value = {"return_code": 0, "blocked": False, "timed_out": True}

        result = run_verification(executor)

        assert result.passed is False

    def test_blocked_command_fails(self):
        executor = MagicMock()
        executor.execute.return_value = {"return_code": -1, "blocked": True, "timed_out": False}

        result = run_verification(executor)

        assert result.passed is False

    def test_custom_command_is_used(self):
        executor = MagicMock()
        executor.execute.return_value = {"return_code": 0, "blocked": False, "timed_out": False}

        result = run_verification(executor, command="make test")

        assert result.command == "make test"
        executor.execute.assert_called_once_with("make test")

    def test_to_dict_shape(self):
        executor = MagicMock()
        executor.execute.return_value = {"return_code": 0, "blocked": False, "timed_out": False}

        result = run_verification(executor)
        d = result.to_dict()

        assert d["passed"] is True
        assert d["command"] == DEFAULT_VERIFY_COMMAND
        assert d["output"]["return_code"] == 0
