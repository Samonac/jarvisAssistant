"""Property tests for Email Client.

Tests Property 16.
"""

import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.config import Config
from app.email_client import EmailClient


def _make_config():
    """Create a minimal config for email client testing."""
    os.environ["LLM_API_KEY"] = "test-key"
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["SMTP_HOST"] = "smtp.example.com"
    os.environ["SMTP_PORT"] = "587"
    os.environ["SMTP_USERNAME"] = "user@example.com"
    os.environ["SMTP_PASSWORD"] = "password123"
    os.environ["SMTP_FROM_ADDRESS"] = "jarvis@example.com"
    config = Config()
    return config


# Feature: jarvis-assistant, Property 16: Email draft composition preserves fields
class TestProperty16:
    """For any valid email fields, composing a draft preserves all fields."""

    @given(
        to=st.from_regex(r"[a-z]{3,10}@[a-z]{3,8}\.[a-z]{2,4}", fullmatch=True),
        subject=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
        body=st.text(min_size=1, max_size=500, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    )
    @settings(max_examples=100)
    def test_draft_composition_preserves_fields(self, to, subject, body):
        config = _make_config()
        client = EmailClient(config)

        draft = client.compose_draft(to, subject, body)

        assert draft.to == to
        assert draft.subject == subject
        assert draft.body == body
        assert draft.from_address == "jarvis@example.com"


class TestEmailNotConfigured:
    """Unit tests for email not configured scenario."""

    def test_is_configured_returns_false_when_missing(self):
        os.environ["LLM_API_KEY"] = "test-key"
        os.environ["LLM_PROVIDER"] = "groq"
        # Clear SMTP settings
        for key in ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_ADDRESS"]:
            os.environ.pop(key, None)
        config = Config()
        client = EmailClient(config)
        assert not client.is_configured()

    def test_send_fails_when_not_configured(self):
        os.environ["LLM_API_KEY"] = "test-key"
        os.environ["LLM_PROVIDER"] = "groq"
        for key in ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_ADDRESS"]:
            os.environ.pop(key, None)
        config = Config()
        client = EmailClient(config)
        draft = client.compose_draft("test@test.com", "Subject", "Body")
        result = client.send(draft)
        assert result["success"] is False
        assert "not configured" in result["message"].lower()
