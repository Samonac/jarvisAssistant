"""Test-gated acceptance for the coding agent (Phase 4-5).

A task's changes are only kept if a verification command succeeds — by
default the project's own pytest suite. This is what a self-reported "done"
must satisfy before it's trusted, per the "no task is considered finished
unless the agent can verify it itself" requirement. apply_verification_gate()
is the shared entry point used by both the interactive /api/agent/* routes
and the autopilot manager.
"""

import logging
from typing import Optional

from app.command_executor import CommandExecutor

logger = logging.getLogger(__name__)

DEFAULT_VERIFY_COMMAND = "python -m pytest -q"


class VerificationResult:
    """Outcome of running a verification command against the working tree."""

    def __init__(self, passed: bool, command: str, output: dict):
        self.passed = passed
        self.command = command
        self.output = output

    def to_dict(self) -> dict:
        return {"passed": self.passed, "command": self.command, "output": self.output}


def run_verification(command_executor: CommandExecutor, command: Optional[str] = None) -> VerificationResult:
    """Run the verification command and report pass/fail.

    A task is considered verified only if the command exits 0 and was
    neither blocked by the command blocklist nor timed out.
    """
    cmd = command or DEFAULT_VERIFY_COMMAND
    result = command_executor.execute(cmd)
    passed = (
        not result.get("blocked")
        and not result.get("timed_out")
        and result.get("return_code") == 0
    )
    if not passed:
        logger.warning("Coding agent verification failed for command '%s': %s", cmd, result)
    return VerificationResult(passed=passed, command=cmd, output=result)


def apply_verification_gate(state, tools, snapshot, verify_command: Optional[str] = None):
    """Gate a self-reported "done" behind a verification command.

    Leaves any other status (awaiting_user, max_iterations_reached, error)
    untouched — only a "done" report is trusted enough to accept or roll back.
    On failure, restores the pre-task snapshot and marks the task rolled_back.

    Args:
        state: An app.coding_agent.state.AgentTaskState.
        tools: An app.coding_agent.tools.CodingAgentTools (for its command_executor).
        snapshot: An app.coding_agent.snapshot.TaskSnapshot keyed by state.session_id.
        verify_command: Optional override for the verification command.
    """
    if state.status != "done":
        return state

    result = run_verification(tools.command_executor, verify_command)
    state.verification = result.to_dict()

    if result.passed:
        snapshot.discard(state.session_id)
        return state

    restored = snapshot.restore(state.session_id)
    state.status = "rolled_back"
    state.error = (
        f"Verification command '{result.command}' failed "
        f"(return_code={result.output.get('return_code')}); changes were rolled back."
        if restored
        else f"Verification command '{result.command}' failed and no snapshot could be restored — "
             f"manual review required."
    )
    return state
