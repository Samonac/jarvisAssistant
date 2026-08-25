"""Tests for the /api/autopilot/* routes (app.autopilot.routes)."""

import json
import os
from unittest.mock import patch

import pytest

from app import create_app
from app.config import Config


@pytest.fixture
def agent_project_dir(tmp_path):
    project_dir = tmp_path / "agent_project"
    project_dir.mkdir()
    (project_dir / "sample.py").write_text("print('hi')\n", encoding="utf-8")
    return project_dir


@pytest.fixture
def app(tmp_path, agent_project_dir):
    env = {
        "LLM_API_KEY": "test_key",
        "LLM_PROVIDER": "groq",
        "DATABASE_PATH": str(tmp_path / "test.db"),
    }
    with patch.dict(os.environ, env, clear=True), patch("app.config.load_dotenv"):
        config = Config()
        flask_app = create_app(config)
        flask_app.config["TESTING"] = True

    patcher = patch("app.autopilot.routes.PROJECT_DIR", str(agent_project_dir))
    patcher.start()
    yield flask_app
    patcher.stop()


@pytest.fixture
def client(app):
    return app.test_client()


class TestStatusAndControls:
    def test_status_reports_disabled_by_default(self, client):
        response = client.get("/api/autopilot/status")
        assert response.status_code == 200
        assert response.get_json()["enabled"] is False

    def test_start_enables_autopilot(self, client):
        response = client.post("/api/autopilot/start")
        assert response.status_code == 200
        assert response.get_json()["enabled"] is True

    def test_pause_disables_autopilot(self, client):
        client.post("/api/autopilot/start")
        response = client.post("/api/autopilot/pause")
        assert response.get_json()["enabled"] is False

    def test_stop_disables_autopilot(self, client):
        client.post("/api/autopilot/start")
        response = client.post("/api/autopilot/stop")
        assert response.get_json()["enabled"] is False


class TestTaskQueueEndpoints:
    def test_add_task_returns_201_and_queues_it(self, client):
        response = client.post(
            "/api/autopilot/tasks", data=json.dumps({"task": "fix the thing"}), content_type="application/json"
        )
        assert response.status_code == 201
        data = response.get_json()
        assert data["task"] == "fix the thing"
        assert data["status"] == "queued"

    def test_add_task_missing_field_returns_400(self, client):
        response = client.post("/api/autopilot/tasks", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400

    def test_list_tasks_returns_added_task(self, client):
        client.post("/api/autopilot/tasks", data=json.dumps({"task": "a task"}), content_type="application/json")
        response = client.get("/api/autopilot/tasks")
        assert response.status_code == 200
        assert len(response.get_json()) == 1

    def test_list_tasks_filters_by_status(self, client):
        client.post("/api/autopilot/tasks", data=json.dumps({"task": "a task"}), content_type="application/json")
        response = client.get("/api/autopilot/tasks?status=done")
        assert response.get_json() == []


class TestRunNow:
    def test_run_now_processes_a_queued_task(self, client):
        client.post("/api/autopilot/tasks", data=json.dumps({"task": "say hi"}), content_type="application/json")

        with patch("app.llm_client.GroqClient.chat", return_value='{"action": "done", "summary": "ok"}'), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            response = client.post("/api/autopilot/run-now")

        assert response.status_code == 200
        assert response.get_json()["status"] == "done"

    def test_run_now_with_no_tasks_reports_reason(self, client):
        response = client.post("/api/autopilot/run-now")
        assert response.get_json() == {"ran": False, "reason": "no_tasks"}


class TestConfirmEndpoint:
    def test_confirm_missing_field_returns_400(self, client):
        client.post("/api/autopilot/tasks", data=json.dumps({"task": "t"}), content_type="application/json")
        response = client.post("/api/autopilot/tasks/1/confirm", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400

    def test_confirm_unknown_task_returns_404(self, client):
        response = client.post(
            "/api/autopilot/tasks/999/confirm",
            data=json.dumps({"accepted": True}),
            content_type="application/json",
        )
        assert response.status_code == 404

    def test_confirm_full_ui_flow(self, client, agent_project_dir):
        (agent_project_dir / "templates").mkdir()
        (agent_project_dir / "templates" / "page.html").write_text("<p>old</p>", encoding="utf-8")

        add_response = client.post(
            "/api/autopilot/tasks", data=json.dumps({"task": "tweak the ui"}), content_type="application/json"
        )
        task_id = add_response.get_json()["id"]

        actions = [
            '{"action": "tool", "tool": "write_file", "args": {"path": "templates/page.html", "content": "<p>new</p>"}}',
            '{"action": "done", "summary": "updated"}',
        ]
        with patch("app.llm_client.GroqClient.chat", side_effect=actions), \
             patch("app.autopilot.manager.apply_verification_gate", side_effect=lambda state, *a, **k: state):
            client.post("/api/autopilot/run-now")

        response = client.post(
            f"/api/autopilot/tasks/{task_id}/confirm",
            data=json.dumps({"accepted": True}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.get_json()["status"] == "confirmed_done"
