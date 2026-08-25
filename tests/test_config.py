"""Property-based tests for Config Manager.

Tests Property 10 and Property 11 from the design document.
"""

import os
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.config import Config, ConfigError, DEFAULT_BLOCKLIST


# Strategy for generating non-empty strings without commas (for blocklist items)
blocklist_pattern_strategy = st.text(
    alphabet=st.characters(
        blacklist_characters=",\x00",
        blacklist_categories=("Cs",),
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() == s and len(s.strip()) > 0)


class TestProperty10MissingEnvVars:
    """Property 10: Missing required env vars produce specific error.

    For any required environment variable that is not set at startup,
    the Config Manager SHALL raise an error whose message contains
    the name of the missing variable.

    **Validates: Requirements 8.3**
    """

    @given(
        missing_var=st.sampled_from(Config.REQUIRED_VARS),
    )
    @settings(max_examples=50)
    def test_missing_required_var_raises_error_with_name(self, missing_var):
        """For any required env var that is missing, ConfigError message contains the var name."""
        # Set up environment with all required vars present EXCEPT the missing one
        env = {}
        for var in Config.REQUIRED_VARS:
            if var != missing_var:
                env[var] = "test_value"

        # Isolate from the real .env file on disk, which would otherwise refill
        # the cleared vars via load_dotenv() and mask the missing-var behavior.
        with patch.dict(os.environ, env, clear=True), patch("app.config.load_dotenv"):
            with pytest.raises(ConfigError) as exc_info:
                Config()

            # The error message must contain the name of the missing variable
            assert missing_var in str(exc_info.value)

    @given(
        missing_vars=st.lists(
            st.sampled_from(Config.REQUIRED_VARS),
            min_size=1,
            max_size=len(Config.REQUIRED_VARS),
            unique=True,
        ),
    )
    @settings(max_examples=50)
    def test_any_subset_of_missing_required_vars_raises_error(self, missing_vars):
        """For any subset of required vars that are missing, at least one is reported."""
        # Set up environment with only the vars NOT in missing_vars
        env = {}
        for var in Config.REQUIRED_VARS:
            if var not in missing_vars:
                env[var] = "test_value"

        # Isolate from the real .env file on disk, which would otherwise refill
        # the cleared vars via load_dotenv() and mask the missing-var behavior.
        with patch.dict(os.environ, env, clear=True), patch("app.config.load_dotenv"):
            with pytest.raises(ConfigError) as exc_info:
                Config()

            # The error message must contain at least one of the missing variable names
            error_msg = str(exc_info.value)
            assert any(var in error_msg for var in missing_vars)


class TestProperty11BlocklistParsingRoundTrip:
    """Property 11: Blocklist parsing round-trip.

    For any list of non-empty pattern strings that do not contain commas,
    joining them with commas and then parsing the result as a COMMAND_BLOCKLIST
    environment variable SHALL produce the original list.

    **Validates: Requirements 8.5**
    """

    @given(
        patterns=st.lists(
            blocklist_pattern_strategy,
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=200)
    def test_blocklist_round_trip(self, patterns):
        """Joining patterns with commas and parsing produces the original list."""
        # Join patterns with commas
        raw = ",".join(patterns)

        # Set up environment with required vars and the blocklist
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
            "COMMAND_BLOCKLIST": raw,
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        # The parsed blocklist should equal the original patterns
        assert config.command_blocklist == patterns

    @given(
        patterns=st.lists(
            blocklist_pattern_strategy,
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=200)
    def test_blocklist_round_trip_preserves_order(self, patterns):
        """Parsing preserves the order of patterns."""
        raw = ",".join(patterns)

        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
            "COMMAND_BLOCKLIST": raw,
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        # Order must be preserved
        for i, pattern in enumerate(patterns):
            assert config.command_blocklist[i] == pattern


class TestConfigDefaults:
    """Unit tests for config default values."""

    def test_default_blocklist_when_env_not_set(self):
        """When COMMAND_BLOCKLIST is not set, default Linux blocklist is used."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        assert config.command_blocklist == DEFAULT_BLOCKLIST

    def test_default_port(self):
        """When PORT is not set, defaults to 5000."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        assert config.port == 5000

    def test_default_command_timeout(self):
        """When COMMAND_TIMEOUT is not set, defaults to 60."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        assert config.command_timeout == 60

    def test_default_scan_timeout(self):
        """When SCAN_TIMEOUT is not set, defaults to 120."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        assert config.scan_timeout == 120

    def test_default_max_history_pairs(self):
        """When MAX_HISTORY_PAIRS is not set, defaults to 10."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        assert config.max_history_pairs == 10

    def test_secret_key_auto_generated(self):
        """When SECRET_KEY is not set, a random key is generated."""
        env = {
            "LLM_API_KEY": "test_key",
            "LLM_PROVIDER": "groq",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Config()

        assert config.secret_key != ""
        assert len(config.secret_key) > 0
