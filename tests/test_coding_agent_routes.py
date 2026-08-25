"""Tests for the /api/agent/* coding-agent routes (app.coding_agent.routes)."""

import json
import os
from unittest.mock import patch

import pytest

from app import create_app
from app.coding_agent.verify import VerificationResult
from app.config import Config


@pytest.fixture
def agent_project_dir(tmp_path):
    """A small fake project tree so snapshot/restore has real files to work with."""
    project_dir = tmp_path / "agent_project"
    project_dir.mkdir()
    (project_dir / "sample.py").write_text("print('hi')\n", encoding="utf-8")
    return project_dir


@pytest.fixture
def app(tmp_path, agent_project_dir):
    """Create a Flask app with test configuration, sandboxed to a throwaway project dir."""
    env = {
        "LLM_API_KEY": "test_key",
        "LLM_PROVIDER": "groq",
        # Keep the lazily-created AgentSessionStore off the real project jarvis.db.
        "DATABASE_PATH": str(tmp_path / "test.db"),
    }
    with patch.dict(os.environ, env, clear=True), patch("app.config.load_dotenv"):
        config = Config()
        flask_app = create_app(config)
        flask_app.config["TESTING"] = True

    # Keep the coding agent (file tools + snapshots) off the real repository.
    project_dir_patcher = patch("app.coding_agent.routes.PROJECT_DIR", str(agent_project_dir))
    project_dir_patcher.start()
    yield flask_app
    project_dir_patcher.stop()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def default_passing_verification():
    """By default, verification passes — individual tests override this to test failure/rollback."""
    with patch("app.coding_agent.verify.run_verification") as mock_verify:
        mock_verify.return_value = VerificationResult(
            passed=True, command="python -m pytest -q", output={"return_code": 0, "stdout": "", "stderr": ""}
        )
        yield mock_verify


class TestAgentRunEndpoint:
    def test_missing_task_returns_400(self, client):
        response = client.post("/api/agent/run", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400
        assert "task" in response.get_json()["error"]

    def test_unknown_provider_returns_400(self, client):
        response = client.post(
            "/api/agent/run",
            data=json.dumps({"task": "do something", "provider": "carrier-pigeon"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_local_provider_returns_400_not_implemented(self, client):
        response = client.post(
            "/api/agent/run",
            data=json.dumps({"task": "do something", "provider": "local"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "local" in response.get_json()["error"].lower()

    def test_successful_run_returns_expected_shape(self, client):
        fake_response = '{"action": "done", "summary": "all done"}'
        with patch("app.llm_client.GroqClient.chat", return_value=fake_response):
            response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "say hi", "effort": "quick"}),
                content_type="application/json",
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "done"
        assert data["summary"] == "all done"
        assert data["session"]["effort"] == "quick"
        assert data["session"]["provider"] == "groq"


class TestAgentSessionStatusEndpoint:
    def test_unknown_session_returns_404(self, client):
        response = client.get("/api/agent/sessions/does-not-exist")
        assert response.status_code == 404

    def test_known_session_returns_its_state(self, client):
        fake_response = '{"action": "done", "summary": "all done"}'
        with patch("app.llm_client.GroqClient.chat", return_value=fake_response):
            run_response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "say hi", "effort": "quick"}),
                content_type="application/json",
            )
        session_id = run_response.get_json()["session_id"]

        response = client.get(f"/api/agent/sessions/{session_id}")
        assert response.status_code == 200
        assert response.get_json()["status"] == "done"


class TestAgentResumeEndpoint:
    def test_missing_fields_returns_400(self, client):
        response = client.post("/api/agent/resume", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400

    def test_unknown_session_returns_404(self, client):
        response = client.post(
            "/api/agent/resume",
            data=json.dumps({"session_id": "nope", "answer": "yes"}),
            content_type="application/json",
        )
        assert response.status_code == 404

    def test_resuming_a_non_paused_session_returns_400(self, client):
        fake_response = '{"action": "done", "summary": "all done"}'
        with patch("app.llm_client.GroqClient.chat", return_value=fake_response):
            run_response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "say hi", "effort": "quick"}),
                content_type="application/json",
            )
        session_id = run_response.get_json()["session_id"]

        response = client.post(
            "/api/agent/resume",
            data=json.dumps({"session_id": session_id, "answer": "yes"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "awaiting" in response.get_json()["error"]

    def test_full_ask_user_then_resume_flow(self, client):
        ask_response = '{"action": "ask_user", "question": "Which file should I touch?"}'
        with patch("app.llm_client.GroqClient.chat", return_value=ask_response):
            run_response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "fix something", "effort": "quick"}),
                content_type="application/json",
            )

        assert run_response.status_code == 200
        run_data = run_response.get_json()
        assert run_data["status"] == "awaiting_user"
        assert run_data["pending_question"] == "Which file should I touch?"
        session_id = run_data["session_id"]

        done_response = '{"action": "done", "summary": "touched config.py"}'
        with patch("app.llm_client.GroqClient.chat", return_value=done_response):
            resume_response = client.post(
                "/api/agent/resume",
                data=json.dumps({"session_id": session_id, "answer": "config.py"}),
                content_type="application/json",
            )

        assert resume_response.status_code == 200
        resume_data = resume_response.get_json()
        assert resume_data["status"] == "done"
        assert resume_data["summary"] == "touched config.py"
        assert resume_data["session_id"] == session_id
        assert resume_data["pending_question"] is None


class TestVerificationGate:
    def test_passed_verification_keeps_status_done(self, client):
        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "ok"}'):
            response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "say hi", "effort": "quick"}),
                content_type="application/json",
            )

        data = response.get_json()
        assert data["status"] == "done"
        assert data["verification"]["passed"] is True

    def test_failed_verification_marks_rolled_back(self, client, default_passing_verification):
        default_passing_verification.return_value = VerificationResult(
            passed=False, command="python -m pytest -q", output={"return_code": 1, "stdout": "", "stderr": "boom"}
        )

        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "ok"}'):
            response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "say hi", "effort": "quick"}),
                content_type="application/json",
            )

        data = response.get_json()
        assert data["status"] == "rolled_back"
        assert data["verification"]["passed"] is False
        assert "rolled back" in data["error"]

    def test_non_done_statuses_are_not_gated(self, client, default_passing_verification):
        """awaiting_user must not trigger verification — the task isn't finished yet."""
        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "ask_user", "question": "which file?"}'):
            response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "say hi", "effort": "quick"}),
                content_type="application/json",
            )

        data = response.get_json()
        assert data["status"] == "awaiting_user"
        assert data["verification"] is None
        default_passing_verification.assert_not_called()

    def test_failed_verification_restores_file_changes_byte_for_byte(self, client, agent_project_dir, default_passing_verification):
        """Integration test: a real file write is rolled back to its exact prior content."""
        original_content = (agent_project_dir / "sample.py").read_text(encoding="utf-8")
        default_passing_verification.return_value = VerificationResult(
            passed=False, command="python -m pytest -q", output={"return_code": 1, "stdout": "", "stderr": "test failed"}
        )

        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "sample.py", "content": "print(\'changed\')\\n"}}',
            '{"action": "done", "summary": "changed sample.py"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions):
            response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "change sample.py", "effort": "standard"}),
                content_type="application/json",
            )

        data = response.get_json()
        assert data["status"] == "rolled_back"
        # The write did happen mid-task...
        assert data["history"][0]["tool"] == "write_file"
        # ...but the working tree was restored to exactly what it was before the task started.
        assert (agent_project_dir / "sample.py").read_text(encoding="utf-8") == original_content


class TestAgentRollbackEndpoint:
    def test_missing_session_id_returns_400(self, client):
        response = client.post("/api/agent/rollback", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400

    def test_unknown_session_returns_404(self, client):
        response = client.post(
            "/api/agent/rollback",
            data=json.dumps({"session_id": "nope"}),
            content_type="application/json",
        )
        assert response.status_code == 404

    def test_manual_rollback_restores_file_and_marks_status(self, client, agent_project_dir):
        original_content = (agent_project_dir / "sample.py").read_text(encoding="utf-8")

        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "sample.py", "content": "print(\'oops\')\\n"}}',
            '{"action": "ask_user", "question": "should I continue?"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions):
            run_response = client.post(
                "/api/agent/run",
                data=json.dumps({"task": "change sample.py", "effort": "standard"}),
                content_type="application/json",
            )
        session_id = run_response.get_json()["session_id"]
        assert (agent_project_dir / "sample.py").read_text(encoding="utf-8") != original_content

        rollback_response = client.post(
            "/api/agent/rollback",
            data=json.dumps({"session_id": session_id}),
            content_type="application/json",
        )

        assert rollback_response.status_code == 200
        assert rollback_response.get_json()["status"] == "rolled_back"
        assert (agent_project_dir / "sample.py").read_text(encoding="utf-8") == original_content
