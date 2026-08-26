"""Unit tests for CodingAgentLoop (app.coding_agent.loop)."""

from unittest.mock import MagicMock
from pathlib import Path

import pytest

from app.agent_session import AgentSessionConfig
from app.coding_agent.loop import CodingAgentLoop
from app.coding_agent.tools import CodingAgentTools


def test_legacy_agent_recovery_prompt_offers_choices():
    from app.agent import AgentExecutor

    prompt = AgentExecutor._recovery_prompt("run_code", "Error:\nSyntaxError")

    assert "[1] I will repair the code and retry it" in prompt
    assert "[2] Retry the original code" in prompt
    assert "[3] Cancel this operation" in prompt


def test_legacy_agent_resolves_vague_run_request_to_last_written_python_file():
    from app.agent import AgentExecutor

    tool_args = AgentExecutor._resolve_run_command(
        {"command": "file"},
        [{"tool": "write_file", "args": {"path": "screenshot_test.py"}}],
    )

    assert tool_args == {"command": 'python "screenshot_test.py"'}


def test_legacy_agent_preserves_explicit_run_command():
    from app.agent import AgentExecutor

    tool_args = AgentExecutor._resolve_run_command(
        {"command": "python other_file.py"},
        [{"tool": "write_file", "args": {"path": "screenshot_test.py"}}],
    )

    assert tool_args == {"command": "python other_file.py"}


def test_legacy_agent_resolves_run_it_anyway_to_last_written_file():
    from app.agent import AgentExecutor

    tool_args = AgentExecutor._resolve_run_command(
        {"command": "it anyway"},
        [{"tool": "write_file", "args": {"path": "app.js"}}],
    )

    assert tool_args == {"command": 'node "app.js"'}


def test_agent_normalizes_unknown_planning_tools():
    from app.agent import AgentExecutor

    plan = AgentExecutor._normalize_plan({
        "plan": True,
        "steps": ["Check the application"],
        "tools_needed": ["windows_session_check", "run_command"],
    })

    assert plan["tools_needed"] == ["run_command"]
    assert plan["unsupported_tools"] == ["windows_session_check"]


def test_tool_output_failure_detection_catches_windows_missing_script():
    from app.agent import AgentExecutor

    output = "Command failed (exit code 2).\ncan't open file 'your_script.py'"

    assert AgentExecutor._tool_output_failed(output)
    assert AgentExecutor._needs_file_discovery(output)


def test_agent_extracts_missing_python_dependency():
    from app.agent import AgentExecutor

    assert AgentExecutor._missing_python_dependency(
        "ModuleNotFoundError: No module named 'pyautogui'"
    ) == "pyautogui"


def test_agent_does_not_repeat_same_tool_action():
    from app.agent import AgentExecutor

    manager = MagicMock()
    manager._execute_tool.return_value = "Command failed (exit code 2)."
    agent = AgentExecutor(manager)
    plan = {"steps": ["run the script"]}
    agent._get_next_action = MagicMock(return_value={
        "tool": "run_command", "args": {"command": "python script.py"}
    })

    result = agent.execute("run the script", "session", [], plan)

    assert result["pending_recovery"]["tool_name"] == "run_command"


def test_conversation_manager_resolves_descriptive_python_filename():
    from app.conversation_manager import ConversationManager

    manager = ConversationManager.__new__(ConversationManager)
    manager.file_manager = MagicMock()
    manager.file_manager.list_directory.return_value = {
        "path": "C:/project",
        "entries": [
            {"name": "screenshot_test.py", "type": "file"},
            {"name": "run.py", "type": "file"},
        ],
    }

    resolved = manager._guess_file_path_from_message(
        "Inspect the screenshot python file"
    )
    assert Path(resolved).name == "screenshot_test.py"


def test_legacy_agent_forces_first_inspection_action_when_model_says_done():
    from app.agent import AgentExecutor

    manager = MagicMock()
    manager._guess_file_path_from_message.return_value = "C:/project/screenshot_test.py"
    agent = AgentExecutor(manager)

    assert agent._first_inspection_action(
        "Inspect the screenshot python file"
    ) == {
        "tool": "read_file",
        "args": {"path": "C:/project/screenshot_test.py"},
    }


@pytest.mark.parametrize(
    ("path", "command"),
    [
        ("script.py", 'python "script.py"'),
        ("script.js", 'node "script.js"'),
        ("script.ts", 'npx tsx "script.ts"'),
        ("script.sh", 'bash "script.sh"'),
        ("script.ps1", 'powershell -NoProfile -ExecutionPolicy Bypass -File "script.ps1"'),
        ("script.bat", '"script.bat"'),
    ],
)
def test_legacy_agent_resolves_common_file_types(path, command):
    from app.agent import AgentExecutor

    tool_args = AgentExecutor._resolve_run_command(
        {"command": "run it"},
        [{"tool": "write_file", "args": {"path": path}}],
    )

    assert tool_args == {"command": command}


class FakeConfig:
    llm_provider = "groq"
    llm_api_key = "primary-key"
    groq_api_key = "groq-key"
    huggingface_api_key = "hf-key"
    gemini_api_key = "gemini-key"


class FakeLLMClient:
    """Returns queued canned responses in order, one per .chat() call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, params=None):
        self.calls.append(messages)
        if not self.responses:
            return '{"action": "done", "summary": "ran out of canned responses"}'
        return self.responses.pop(0)


def make_loop(responses, effort="standard", tools=None):
    session = AgentSessionConfig(FakeConfig(), effort=effort)
    session.llm_client = FakeLLMClient(responses)
    agent_tools = tools or CodingAgentTools(MagicMock(), MagicMock(), backup_dir="/tmp/unused")
    return CodingAgentLoop(session, agent_tools, project_dir="/project"), session


class TestImmediateDone:
    def test_done_on_first_response(self):
        loop, _ = make_loop(['{"action": "done", "summary": "nothing to do"}'])
        state = loop.start("do nothing")

        assert state.status == "done"
        assert state.summary == "nothing to do"
        assert state.history == []
        assert state.iteration == 1


class TestToolCallThenDone:
    def test_records_tool_call_and_finishes(self):
        file_manager = MagicMock()
        file_manager.read_file.return_value = {"content": "print('hi')", "path": "a.py"}
        tools = CodingAgentTools(file_manager, MagicMock(), backup_dir="/tmp/unused")

        responses = [
            '{"action": "tool", "tool": "read_file", "args": {"path": "a.py"}, "thought": "check contents"}',
            '{"action": "done", "summary": "read the file"}',
        ]
        loop, _ = make_loop(responses, tools=tools)
        state = loop.start("read a.py")

        assert state.status == "done"
        assert len(state.history) == 1
        assert state.history[0].tool == "read_file"
        assert state.history[0].output["content"] == "print('hi')"

    def test_unknown_tool_name_produces_error_observation_but_continues(self):
        responses = [
            '{"action": "tool", "tool": "fly_to_moon", "args": {}}',
            '{"action": "done", "summary": "gave up on the moon"}',
        ]
        loop, _ = make_loop(responses)
        state = loop.start("fly to the moon")

        assert state.status == "done"
        assert "error" in state.history[0].output


class TestInvalidJsonHandling:
    def test_single_retry_then_success(self):
        responses = [
            "not json at all",
            '{"action": "done", "summary": "recovered"}',
        ]
        loop, session = make_loop(responses)
        state = loop.start("task")

        assert state.status == "done"
        assert state.summary == "recovered"
        assert session.llm_client.calls  # confirm retry prompt was actually sent

    def test_invalid_json_twice_results_in_error_status(self):
        responses = ["still not json", "still not json either"]
        loop, _ = make_loop(responses)
        state = loop.start("task")

        assert state.status == "error"
        assert "valid action" in state.error

    def test_json_wrapped_in_markdown_fence_is_parsed(self):
        responses = ['```json\n{"action": "done", "summary": "fenced"}\n```']
        loop, _ = make_loop(responses)
        state = loop.start("task")

        assert state.status == "done"
        assert state.summary == "fenced"


class TestIterationBudget:
    def test_stops_at_max_iterations_for_quick_tier(self):
        file_manager = MagicMock()
        file_manager.read_file.return_value = {"content": "x"}
        tools = CodingAgentTools(file_manager, MagicMock(), backup_dir="/tmp/unused")

        # Always asks to call a tool, never says done — should be cut off by the budget.
        responses = ['{"action": "tool", "tool": "read_file", "args": {"path": "a.py"}}'] * 10
        loop, session = make_loop(responses, effort="quick", tools=tools)
        state = loop.start("infinite task")

        assert state.status == "max_iterations_reached"
        assert state.iteration == session.max_iterations == 3
        assert len(state.history) == 3


class TestTimeBudget:
    def test_past_deadline_stops_immediately_without_calling_the_llm(self):
        from datetime import datetime, timedelta

        loop, session = make_loop(['{"action": "done", "summary": "should never be reached"}'])
        past_deadline = datetime.now() - timedelta(seconds=1)

        state = loop.start("some task", deadline=past_deadline)

        assert state.status == "time_budget_exceeded"
        assert state.iteration == 0
        assert session.llm_client.calls == []

    def test_no_deadline_means_no_time_limit(self):
        """Passing no deadline preserves existing (Phase 2-4) unbounded-by-time behavior."""
        loop, _ = make_loop(['{"action": "done", "summary": "ok"}'])
        state = loop.start("some task", deadline=None)
        assert state.status == "done"


class TestAskUserPauseAndResume:
    def test_ask_user_pauses_with_pending_question(self):
        responses = ['{"action": "ask_user", "question": "Which file should I edit?"}']
        loop, _ = make_loop(responses)
        state = loop.start("edit the config")

        assert state.status == "awaiting_user"
        assert state.pending_question == "Which file should I edit?"
        assert state.iteration == 1

    def test_resume_folds_answer_into_same_transcript_and_continues(self):
        ask_response = '{"action": "ask_user", "question": "Which file should I edit?"}'
        loop, session = make_loop([ask_response])
        state = loop.start("edit the config")
        assert state.status == "awaiting_user"

        # Queue up what the LLM should say once it has the user's answer.
        session.llm_client.responses.append('{"action": "done", "summary": "edited config.py"}')

        resumed = loop.resume(state, "config.py")

        assert resumed.status == "done"
        assert resumed.summary == "edited config.py"
        # Same session/task carried through, not a fresh one.
        assert resumed.session_id == state.session_id
        # The answer was folded in as a reply to the specific pending question.
        assert resumed.pending_question is None
        last_user_message = [m for m in resumed.messages if m["role"] == "user"][-1]
        assert "Which file should I edit?" in last_user_message["content"]
        assert "config.py" in last_user_message["content"]

    def test_resume_continues_iteration_budget_rather_than_resetting(self):
        ask_response = '{"action": "ask_user", "question": "Proceed?"}'
        loop, session = make_loop([ask_response], effort="quick")
        state = loop.start("do a risky thing")
        assert state.iteration == 1

        session.llm_client.responses.append('{"action": "done", "summary": "done"}')
        resumed = loop.resume(state, "yes")

        assert resumed.iteration == 2  # continued counting, did not reset to 1

    def test_resuming_a_non_paused_session_raises(self):
        loop, _ = make_loop(['{"action": "done", "summary": "already finished"}'])
        state = loop.start("quick task")
        assert state.status == "done"

        with pytest.raises(ValueError, match="awaiting_user"):
            loop.resume(state, "some answer")
