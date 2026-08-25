"""Task/session state for the interactive coding agent (Phases 2-3)."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ToolCallRecord:
    """One tool call and its observed result."""

    tool: str
    args: dict
    output: dict
    thought: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class AgentTaskState:
    """Full transcript and outcome of one coding-agent run.

    `messages` holds the raw LLM conversation transcript (system/user/
    assistant turns) needed to resume the loop exactly where it paused —
    it is persisted (app.coding_agent.session_store.AgentSessionStore) but
    deliberately excluded from the public `to_dict()` API response.
    """

    session_id: str
    task: str
    status: str = "in_progress"  # in_progress | awaiting_user | done | rolled_back | max_iterations_reached | time_budget_exceeded | error
    iteration: int = 0
    history: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    pending_question: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    verification: Optional[dict] = None
    # Session config snapshot, needed to rebuild the LLMClient/tools on resume.
    provider: str = ""
    model: Optional[str] = None
    effort: str = "standard"

    def to_dict(self) -> dict:
        """Public, API-facing representation (excludes the raw LLM transcript)."""
        return {
            "session_id": self.session_id,
            "task": self.task,
            "status": self.status,
            "iteration": self.iteration,
            "pending_question": self.pending_question,
            "summary": self.summary,
            "error": self.error,
            "verification": self.verification,
            "history": [
                {
                    "tool": h.tool,
                    "args": h.args,
                    "output": h.output,
                    "thought": h.thought,
                    "timestamp": h.timestamp,
                }
                for h in self.history
            ],
        }
