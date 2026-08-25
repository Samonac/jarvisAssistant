"""Interactive coding-agent loop (Phases 2-3): plan -> act -> observe -> iterate.

No autonomy/nightly behavior here — every run/resume executes synchronously
within a single request, bounded by the session's thinking-effort iteration
budget (and an optional wall-clock deadline, used by the Phase 5 autopilot
mode). The loop can pause mid-task via an `ask_user` action (Phase 3): the
full transcript is captured on AgentTaskState so a caller can persist it
(app.coding_agent.session_store.AgentSessionStore) and later call `resume()`
with the user's answer folded back into the exact same plan. No git/snapshot
rollback yet (Phase 4) — file/delete tool calls are backed up on disk by
app.coding_agent.tools.CodingAgentTools but not yet auto-restored.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Optional

from app.agent_session import AgentSessionConfig
from app.coding_agent.prompts import CODING_AGENT_SYSTEM_PROMPT, INVALID_JSON_RETRY_PROMPT, format_user_answer
from app.coding_agent.state import AgentTaskState, ToolCallRecord
from app.coding_agent.tools import CodingAgentTools

logger = logging.getLogger(__name__)

MAX_JSON_RETRIES = 1


class CodingAgentLoop:
    """Runs a coding task using a bounded plan/act/observe loop, pausable via ask_user."""

    def __init__(self, session: AgentSessionConfig, tools: CodingAgentTools, project_dir: str):
        self.session = session
        self.tools = tools
        self.project_dir = project_dir

    def start(self, task: str, session_id: Optional[str] = None, deadline: Optional[datetime] = None) -> AgentTaskState:
        """Begin a brand-new task and run until it pauses, finishes, or errors.

        Args:
            task: The user's instruction.
            session_id: Optional pre-generated session id. Pass this in when a
                caller needs the id up front (e.g. to snapshot the project
                tree) before any tool call can happen.
            deadline: Optional wall-clock cutoff. If reached, the loop stops
                with status "time_budget_exceeded" regardless of remaining
                iteration budget. Used by the autopilot mode to bound how long
                an unattended task may run.
        """
        state = AgentTaskState(
            session_id=session_id or uuid.uuid4().hex,
            task=task,
            provider=self.session.provider,
            model=self.session.model,
            effort=self.session.effort.name,
            messages=[
                {"role": "system", "content": CODING_AGENT_SYSTEM_PROMPT.format(project_dir=self.project_dir)},
                {"role": "user", "content": task},
            ],
        )
        return self._run(state, deadline=deadline)

    def resume(self, state: AgentTaskState, answer: str, deadline: Optional[datetime] = None) -> AgentTaskState:
        """Resume a previously paused (awaiting_user) session with the user's answer."""
        if state.status != "awaiting_user":
            raise ValueError(
                f"Cannot resume session '{state.session_id}' with status '{state.status}' "
                "(expected 'awaiting_user')."
            )

        question = state.pending_question or ""
        state.messages.append({"role": "user", "content": format_user_answer(question, answer)})
        state.pending_question = None
        state.status = "in_progress"
        return self._run(state, deadline=deadline)

    def _run(self, state: AgentTaskState, deadline: Optional[datetime] = None) -> AgentTaskState:
        """Advance the loop from state.iteration until it pauses, finishes, or errors."""
        messages = state.messages

        while state.iteration < self.session.max_iterations:
            if deadline is not None and datetime.now() >= deadline:
                state.status = "time_budget_exceeded"
                state.summary = "Stopped after exceeding the task's time budget."
                return state

            state.iteration += 1
            action = self._get_next_action(messages)

            if action is None:
                state.status = "error"
                state.error = "The model did not return a valid action after retrying."
                return state

            action_type = action.get("action")

            if action_type == "done":
                state.status = "done"
                state.summary = action.get("summary", "Task completed.")
                messages.append({"role": "assistant", "content": json.dumps(action)})
                return state

            if action_type == "ask_user":
                question = (action.get("question") or "").strip() or "Could you clarify what you'd like me to do?"
                state.status = "awaiting_user"
                state.pending_question = question
                messages.append({"role": "assistant", "content": json.dumps(action)})
                return state

            tool_name = action.get("tool", "")
            args = action.get("args") or {}
            thought = action.get("thought", "")

            if tool_name not in CodingAgentTools.TOOL_NAMES:
                observation = {"error": f"Unknown tool: '{tool_name}'. Available: {', '.join(sorted(CodingAgentTools.TOOL_NAMES))}"}
            else:
                observation = self.tools.dispatch(tool_name, args)

            state.history.append(ToolCallRecord(tool=tool_name, args=args, output=observation, thought=thought))

            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({
                "role": "user",
                "content": f"Tool result for '{tool_name}': {json.dumps(observation)[:4000]}\n\nWhat is the next action?",
            })

        state.status = "max_iterations_reached"
        state.summary = (
            f"Stopped after reaching the {self.session.max_iterations}-iteration budget "
            f"for effort tier '{self.session.effort.name}'."
        )
        return state

    def _get_next_action(self, messages: list[dict]) -> Optional[dict]:
        """Ask the LLM for the next action, retrying once on invalid JSON."""
        for attempt in range(MAX_JSON_RETRIES + 1):
            response = self.session.llm_client.chat(messages, params=self.session.inference_params)
            parsed = self._parse_json_action(response)
            if parsed is not None:
                return parsed

            logger.warning("Coding agent: invalid JSON action (attempt %d): %r", attempt + 1, response[:200])
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": INVALID_JSON_RETRY_PROMPT})

        return None

    @staticmethod
    def _parse_json_action(response: str) -> Optional[dict]:
        """Extract a single JSON action object from the model's response."""
        stripped = response.strip()

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
        candidate = fenced.group(1) if fenced else stripped

        data = CodingAgentLoop._try_load_json(candidate)
        if data is None:
            brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if brace_match:
                data = CodingAgentLoop._try_load_json(brace_match.group(0))

        if not isinstance(data, dict) or "action" not in data:
            return None
        return data

    @staticmethod
    def _try_load_json(text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
