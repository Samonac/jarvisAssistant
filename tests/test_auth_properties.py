"""Property tests for Authentication Manager.

Tests Properties 20 and 21.
"""

import os

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from flask import Flask

from app.config import Config
from app.auth import AuthManager


def _make_config(username="admin", password="secret123"):
    os.environ["LLM_API_KEY"] = "test-key"
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["WEB_USERNAME"] = username
    os.environ["WEB_PASSWORD"] = password
    return Config()


# Feature: jarvis-assistant, Property 20: Authentication rejects invalid credentials
class TestProperty20:
    """For any credentials that don't match, authentication returns False."""

    @given(
        username=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N"))),
        password=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=100)
    def test_rejects_invalid_credentials(self, username, password):
        # Ensure the generated credentials don't accidentally match
        assume(username != "admin" or password != "secret123")

        config = _make_config("admin", "secret123")
        auth = AuthManager(config)

        assert auth.authenticate(username, password) is False


# Feature: jarvis-assistant, Property 21: Authenticated sessions grant access
class TestProperty21:
    """After successful authentication, the session grants access."""

    def test_valid_credentials_authenticate(self):
        config = _make_config("admin", "secret123")
        auth = AuthManager(config)
        assert auth.authenticate("admin", "secret123") is True

    def test_session_grants_access(self):
        config = _make_config("admin", "secret123")
        auth = AuthManager(config)

        app = Flask(__name__)
        app.secret_key = "test-secret"

        with app.test_request_context():
            from flask import session
            auth.create_session("admin")
            assert auth.is_authenticated() is True

    def test_no_session_denies_access(self):
        config = _make_config("admin", "secret123")
        auth = AuthManager(config)

        app = Flask(__name__)
        app.secret_key = "test-secret"

        with app.test_request_context():
            assert auth.is_authenticated() is False

    def test_destroyed_session_denies_access(self):
        config = _make_config("admin", "secret123")
        auth = AuthManager(config)

        app = Flask(__name__)
        app.secret_key = "test-secret"

        with app.test_request_context():
            from flask import session
            auth.create_session("admin")
            assert auth.is_authenticated() is True
            auth.destroy_session()
            assert auth.is_authenticated() is False
