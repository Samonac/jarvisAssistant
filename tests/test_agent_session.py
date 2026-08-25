"""Unit tests for coding-agent session configuration (app.agent_session)."""

import pytest
from unittest.mock import patch

from app.agent_session import AgentSessionConfig
from app.llm_client import GroqClient, HuggingFaceClient, GeminiClient


def make_config(**overrides):
    """Build a minimal Config-like stand-in with just the attributes AgentSessionConfig reads."""

    class FakeConfig:
        llm_provider = "groq"
        llm_api_key = "primary-key"
        groq_api_key = "groq-key"
        huggingface_api_key = "hf-key"
        gemini_api_key = "gemini-key"

    cfg = FakeConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestAgentSessionConfig:
    def test_defaults_to_config_provider_and_standard_effort(self):
        session = AgentSessionConfig(make_config())
        assert session.provider == "groq"
        assert isinstance(session.llm_client, GroqClient)
        assert session.effort.name == "standard"
        assert session.max_iterations == 8

    @pytest.mark.parametrize(
        "provider,expected_cls",
        [("groq", GroqClient), ("huggingface", HuggingFaceClient), ("gemini", GeminiClient)],
    )
    def test_provider_selection_builds_correct_client(self, provider, expected_cls):
        session = AgentSessionConfig(make_config(), provider=provider)
        assert isinstance(session.llm_client, expected_cls)

    def test_model_override_is_applied_to_client(self):
        session = AgentSessionConfig(make_config(), provider="groq", model="custom-model")
        assert session.llm_client.model == "custom-model"

    def test_effort_tier_drives_inference_params(self):
        session = AgentSessionConfig(make_config(), effort="deep")
        assert session.inference_params.max_tokens == 2048
        assert session.max_iterations == 20

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            AgentSessionConfig(make_config(), provider="carrier-pigeon")

    def test_local_provider_is_recognized_but_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AgentSessionConfig(make_config(), provider="local")

    def test_missing_api_key_raises_clear_error(self):
        cfg = make_config(gemini_api_key=None, llm_provider="groq")
        with pytest.raises(ValueError, match="No API key configured"):
            AgentSessionConfig(cfg, provider="gemini")

    def test_to_dict_reports_resolved_session(self):
        session = AgentSessionConfig(make_config(), provider="groq", model="foo", effort="quick")
        d = session.to_dict()
        assert d == {
            "provider": "groq",
            "model": "foo",
            "effort": "quick",
            "max_iterations": 3,
            "max_tokens": 512,
        }
