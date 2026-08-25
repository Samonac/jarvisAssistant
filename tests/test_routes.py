"""Tests for Flask server routes.

Property-based tests (Property 1) and unit tests for health endpoint and startup logging.
"""

import json
import logging
import os
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app import create_app
from app.auth import AuthManager
from app.config import Config


# --- Fixtures ---


@pytest.fixture
def app():
    """Create a Flask app with test configuration."""
    env = {
        "LLM_API_KEY": "test_key",
        "LLM_PROVIDER": "groq",
    }
    # Isolate from the real .env file on disk (which sets WEB_PASSWORD), so
    # auth stays disabled here as the explicit env dict above intends.
    with patch.dict(os.environ, env, clear=True), patch("app.config.load_dotenv"):
        config = Config()
        app = create_app(config)
        app.config["TESTING"] = True
        # WEB_PASSWORD is unset above, so AuthManager runs in disabled mode
        # (is_authenticated() always True) — mirrors production wiring in run.py.
        app.config["AUTH_MANAGER"] = AuthManager(config)
        return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


# --- Property 1: Invalid requests are rejected ---


class TestProperty1InvalidRequestsRejected:
    """Property 1: Invalid requests are rejected.

    For any JSON request body that does not contain a `message` field,
    the `/chat` endpoint SHALL return a 400 status code and the response
    body SHALL contain a descriptive error message.

    **Validates: Requirements 1.3**
    """

    @given(
        body=st.dictionaries(
            keys=st.text(min_size=1, max_size=30).filter(lambda k: k != "message"),
            values=st.one_of(
                st.text(max_size=50),
                st.integers(),
                st.booleans(),
                st.none(),
            ),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_missing_message_field_returns_400(self, body):
        """Any JSON body without 'message' key returns 400 with error description."""
        # Ensure 'message' is not in the body
        assume("message" not in body)

        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            app = create_app(config)
            app.config["TESTING"] = True
            client = app.test_client()

            response = client.post(
                "/chat",
                data=json.dumps(body),
                content_type="application/json",
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert len(data["error"]) > 0

    @given(
        content_type=st.sampled_from(
            ["text/plain", "text/html", "application/xml", ""]
        ),
    )
    @settings(max_examples=20)
    def test_non_json_content_type_returns_400(self, content_type):
        """Non-JSON content types that can't be parsed return 400."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            app = create_app(config)
            app.config["TESTING"] = True
            client = app.test_client()

            response = client.post(
                "/chat",
                data="not json",
                content_type=content_type,
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data

    @given(
        extra_fields=st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(lambda k: k != "message"),
            values=st.text(max_size=30),
            min_size=0,
            max_size=3,
        ),
    )
    @settings(max_examples=50)
    def test_valid_message_with_extra_fields_returns_200(self, extra_fields):
        """A body with 'message' field (plus any extras) returns 200."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            app = create_app(config)
            app.config["TESTING"] = True
            client = app.test_client()

            body = {"message": "Hello Jarvis", **extra_fields}
            response = client.post(
                "/chat",
                data=json.dumps(body),
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert "response" in data
            assert "session_id" in data


# --- Unit Tests: Health Endpoint ---


class TestHealthEndpoint:
    """Unit tests for the GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_structure(self, client):
        """Health endpoint returns status, provider, and capabilities."""
        response = client.get("/health")
        data = response.get_json()

        assert "status" in data
        assert "provider" in data
        assert "capabilities" in data
        assert data["status"] == "ok"

    def test_health_returns_configured_provider(self, client):
        """Health endpoint returns the configured LLM provider."""
        response = client.get("/health")
        data = response.get_json()
        assert data["provider"] == "groq"

    def test_health_capabilities_is_list(self, client):
        """Health endpoint capabilities is a list of strings."""
        response = client.get("/health")
        data = response.get_json()
        assert isinstance(data["capabilities"], list)
        assert all(isinstance(c, str) for c in data["capabilities"])

    def test_health_default_capabilities(self, client):
        """Health endpoint includes default capabilities."""
        response = client.get("/health")
        data = response.get_json()
        assert "chat" in data["capabilities"]
        assert "command_execution" in data["capabilities"]
        assert "web_search" in data["capabilities"]
        assert "network_scan" in data["capabilities"]

    def test_health_with_calendar_configured(self):
        """Health endpoint includes calendar when configured."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "gemini",
            "CALENDAR_PROVIDER": "google",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            app = create_app(config)
            app.config["TESTING"] = True
            client = app.test_client()

            response = client.get("/health")
            data = response.get_json()
            assert "calendar" in data["capabilities"]

    def test_health_with_email_configured(self):
        """Health endpoint includes email when SMTP is configured."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "huggingface",
            "SMTP_HOST": "smtp.example.com",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            app = create_app(config)
            app.config["TESTING"] = True
            client = app.test_client()

            response = client.get("/health")
            data = response.get_json()
            assert "email" in data["capabilities"]


# --- Unit Tests: Startup Logging ---


class TestStartupLogging:
    """Unit tests for startup logging of provider and capabilities."""

    def test_startup_logs_provider(self, caplog):
        """App creation logs the configured LLM provider."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            with caplog.at_level(logging.INFO, logger="app"):
                create_app(config)

            assert any("groq" in record.message for record in caplog.records)

    def test_startup_logs_capabilities(self, caplog):
        """App creation logs available capabilities."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "gemini",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            with caplog.at_level(logging.INFO, logger="app"):
                create_app(config)

            log_messages = " ".join(record.message for record in caplog.records)
            assert "chat" in log_messages
            assert "gemini" in log_messages

    def test_startup_logs_with_all_capabilities(self, caplog):
        """App creation logs all capabilities when fully configured."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
            "CALENDAR_PROVIDER": "caldav",
            "SMTP_HOST": "smtp.example.com",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config()
            with caplog.at_level(logging.INFO, logger="app"):
                create_app(config)

            log_messages = " ".join(record.message for record in caplog.records)
            assert "calendar" in log_messages
            assert "email" in log_messages


# --- Unit Tests: Chat Endpoint ---


class TestChatEndpoint:
    """Unit tests for the POST /chat endpoint."""

    def test_valid_chat_returns_200(self, client):
        """Valid chat request returns 200."""
        response = client.post(
            "/chat",
            data=json.dumps({"message": "Hello"}),
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_valid_chat_returns_response_and_session(self, client):
        """Valid chat returns response text and session_id."""
        response = client.post(
            "/chat",
            data=json.dumps({"message": "Hello"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert "response" in data
        assert "session_id" in data
        assert len(data["response"]) > 0
        assert len(data["session_id"]) > 0

    def test_chat_preserves_provided_session_id(self, client):
        """Chat endpoint uses provided session_id."""
        response = client.post(
            "/chat",
            data=json.dumps({"message": "Hello", "session_id": "my-session-123"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["session_id"] == "my-session-123"

    def test_chat_generates_session_id_when_missing(self, client):
        """Chat endpoint generates a session_id when not provided."""
        response = client.post(
            "/chat",
            data=json.dumps({"message": "Hello"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

    def test_chat_empty_body_returns_400(self, client):
        """Empty request body returns 400."""
        response = client.post(
            "/chat",
            data="",
            content_type="application/json",
        )
        assert response.status_code == 400


# --- Unit Tests: API Endpoints ---


class TestAPIEndpoints:
    """Unit tests for API configuration, metrics, and system endpoints."""

    def test_get_config_returns_200(self, client):
        """GET /api/config returns 200."""
        response = client.get("/api/config")
        assert response.status_code == 200

    def test_get_config_returns_expected_fields(self, client):
        """GET /api/config returns expected configuration fields."""
        response = client.get("/api/config")
        data = response.get_json()
        assert "provider" in data
        assert "port" in data
        assert "command_timeout" in data
        assert "scan_timeout" in data
        assert "max_history_pairs" in data

    def test_get_config_does_not_expose_secrets(self, client):
        """GET /api/config does not expose API keys or passwords."""
        response = client.get("/api/config")
        data = response.get_json()
        assert "llm_api_key" not in data
        assert "web_password" not in data
        assert "secret_key" not in data
        assert "smtp_password" not in data

    def test_put_config_valid_update(self, client):
        """PUT /api/config with valid key/value returns success."""
        response = client.put(
            "/api/config",
            data=json.dumps({"key": "command_timeout", "value": 30}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

    def test_put_config_invalid_key(self, client):
        """PUT /api/config with non-updatable key returns 400."""
        response = client.put(
            "/api/config",
            data=json.dumps({"key": "llm_api_key", "value": "hacked"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_put_config_invalid_value(self, client):
        """PUT /api/config with invalid value returns 400."""
        response = client.put(
            "/api/config",
            data=json.dumps({"key": "command_timeout", "value": -5}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_get_metrics_returns_200(self, client):
        """GET /api/metrics returns 200."""
        response = client.get("/api/metrics")
        assert response.status_code == 200

    def test_get_metrics_returns_expected_fields(self, client):
        """GET /api/metrics returns expected metric fields."""
        response = client.get("/api/metrics")
        data = response.get_json()
        assert "total_calls" in data
        assert "calls_today" in data
        assert "avg_response_ms" in data
        assert "tool_usage" in data
        assert "error_rate" in data

    def test_get_system_returns_200(self, client):
        """GET /api/system returns 200."""
        response = client.get("/api/system")
        assert response.status_code == 200

    def test_get_system_returns_expected_fields(self, client):
        """GET /api/system returns expected system fields."""
        response = client.get("/api/system")
        data = response.get_json()
        assert "cpu_percent" in data
        assert "ram_used_mb" in data
        assert "ram_total_mb" in data
        assert "ram_percent" in data
        assert "disk_used_gb" in data
        assert "disk_total_gb" in data
        assert "disk_percent" in data


# --- Unit Tests: Frontend Routes ---


class TestFrontendRoutes:
    """Unit tests for frontend placeholder routes."""

    def test_index_returns_200(self, client):
        """GET / returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_dashboard_returns_200(self, client):
        """GET /dashboard returns 200."""
        response = client.get("/dashboard")
        assert response.status_code == 200

    def test_settings_returns_200(self, client):
        """GET /settings returns 200."""
        response = client.get("/settings")
        assert response.status_code == 200
