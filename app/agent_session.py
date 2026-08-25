"""Coding-agent session configuration (Phase 1).

Resolves the user's choice of LLM provider/model and thinking-effort tier,
made once at the start of a coding-agent session, into a ready-to-use
LLMClient + InferenceParams + iteration budget. This is deliberately
independent of app.conversation_manager's Jarvis chat persona/tools —
per the project decision, the coding agent is a separate interaction mode.
"""

import logging

from app.config import Config
from app.effort import DEFAULT_EFFORT, EffortConfig, get_effort_config
from app.llm_client import InferenceParams, LLMClient, create_llm_client

logger = logging.getLogger(__name__)

# Providers that don't require an API key lookup in Config (e.g. local inference).
KEYLESS_PROVIDERS = {"local", "local_llm"}


class AgentSessionConfig:
    """Resolved configuration for a single coding-agent session.

    Attributes:
        provider: LLM provider name ("groq", "huggingface", "gemini", or "local").
        model: Optional explicit model name/path override for the chosen provider.
        effort: The EffortConfig derived from the requested thinking-effort tier.
        llm_client: The instantiated LLMClient for this session.
        inference_params: Per-session InferenceParams derived from the effort tier
            (isolated from the global `app.llm_client.inference_params` singleton
            used by the existing Jarvis chat mode).
    """

    def __init__(
        self,
        config: Config,
        provider: str = None,
        model: str = None,
        effort: str = DEFAULT_EFFORT,
    ):
        self.provider = (provider or config.llm_provider).lower().strip()
        self.model = model
        self.effort: EffortConfig = get_effort_config(effort)

        api_key = self._resolve_api_key(config, self.provider)
        self.llm_client: LLMClient = create_llm_client(self.provider, api_key, model=self.model)

        self.inference_params = InferenceParams(
            temperature=self.effort.temperature,
            max_tokens=self.effort.max_tokens,
        )

        logger.info(
            "Agent session configured: provider=%s model=%s effort=%s max_iterations=%d",
            self.provider, self.model or "(default)", self.effort.name, self.effort.max_iterations,
        )

    @staticmethod
    def _resolve_api_key(config: Config, provider: str) -> str:
        """Look up the configured API key for the requested provider.

        Returns an empty string for keyless providers (e.g. "local").
        """
        if provider in KEYLESS_PROVIDERS:
            return ""

        keys_by_provider = {
            "groq": config.groq_api_key,
            "huggingface": config.huggingface_api_key,
            "gemini": config.gemini_api_key,
        }
        key = keys_by_provider.get(provider)
        if not key and provider == config.llm_provider.lower().strip():
            key = config.llm_api_key
        if not key:
            raise ValueError(f"No API key configured for provider '{provider}'.")
        return key

    @property
    def max_iterations(self) -> int:
        return self.effort.max_iterations

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort.name,
            "max_iterations": self.effort.max_iterations,
            "max_tokens": self.effort.max_tokens,
        }
