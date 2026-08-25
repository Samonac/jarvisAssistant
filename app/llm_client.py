"""LLM Client module for Jarvis Assistant.

Provides an abstract interface for LLM providers with concrete implementations
for Groq, HuggingFace, and Google Gemini free-tier APIs.

Uses streaming where supported to minimize peak memory usage on the Raspberry Pi.
All errors are caught and returned as fallback messages — no unhandled exceptions.
"""

import logging
import socket
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Fallback messages for different error scenarios
FALLBACK_CONNECTION_ERROR = (
    "I'm afraid my connection to the neural network is temporarily disrupted, Sir. "
    "Please try again momentarily."
)
FALLBACK_RATE_LIMIT = (
    "I must ask for your patience, Sir. The service requires a brief respite "
    "before I can process further requests."
)
FALLBACK_GENERIC_ERROR = (
    "I regret to inform you that the service is temporarily unavailable, Sir. "
    "Please try again at a later date."
)
FALLBACK_NETWORK_ERROR = (
    "It appears we've lost network connectivity, Sir. "
    "Please verify your WiFi connection and try again."
)

# Request timeout in seconds (generous for Pi's WiFi latency)
REQUEST_TIMEOUT = 30


class InferenceParams:
    """Configurable inference parameters for LLM calls.

    Attributes:
        temperature: Controls randomness (0.0 = deterministic, 2.0 = very random). Default 0.7.
        top_p: Nucleus sampling threshold. Default 0.9.
        max_tokens: Maximum tokens to generate. Default 1024.
        frequency_penalty: Penalize repeated tokens (-2.0 to 2.0). Default 0.0.
        presence_penalty: Penalize new topics (-2.0 to 2.0). Default 0.0.
    """

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def to_dict(self) -> dict:
        """Return params as a dict for API payloads."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


# Global inference params instance — modified at runtime via the UI
inference_params = InferenceParams()


class LLMClient(ABC):
    """Abstract base class for LLM provider clients.

    All implementations must handle errors gracefully and return
    fallback messages rather than raising unhandled exceptions.
    """

    @abstractmethod
    def chat(self, messages: list[dict], params: Optional["InferenceParams"] = None) -> str:
        """Send messages and return generated text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            params: Optional per-call inference parameters. Defaults to the
                shared module-level `inference_params` when omitted.

        Returns:
            Generated text response string. On error, returns a fallback message.
        """
        pass


class GroqClient(LLMClient):
    """LLM client using Groq API with streaming support.

    Uses the Groq OpenAI-compatible chat completions endpoint.
    Model: llama-3.3-70b-versatile (free tier).
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.MODEL

    def chat(self, messages: list[dict], params: Optional[InferenceParams] = None) -> str:
        """Send messages to Groq API with streaming and return generated text."""
        params = params or inference_params
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_tokens": params.max_tokens,
                "frequency_penalty": params.frequency_penalty,
                "presence_penalty": params.presence_penalty,
            }

            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                logger.warning("Groq API rate limit exceeded")
                return FALLBACK_RATE_LIMIT

            if response.status_code >= 400:
                logger.error(
                    "Groq API error: %d %s", response.status_code, response.text[:200]
                )
                return FALLBACK_GENERIC_ERROR

            # Stream response chunks and accumulate content
            content = []
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]  # Strip "data: " prefix
                if data.strip() == "[DONE]":
                    break
                try:
                    import json

                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        content.append(delta["content"])
                except (ValueError, KeyError, IndexError):
                    continue

            result = "".join(content)
            if not result:
                logger.warning("Groq API returned empty response")
                return FALLBACK_GENERIC_ERROR
            return result

        except requests.exceptions.Timeout as e:
            logger.error("Timeout error (Groq): %s", e)
            return FALLBACK_CONNECTION_ERROR
        except requests.exceptions.ConnectionError as e:
            logger.error("Connection error (Groq): %s", e)
            return FALLBACK_NETWORK_ERROR
        except (socket.gaierror, OSError) as e:
            logger.error("Network connectivity error (Groq): %s", e)
            return FALLBACK_NETWORK_ERROR
        except requests.exceptions.RequestException as e:
            logger.error("Request error (Groq): %s", e)
            return FALLBACK_GENERIC_ERROR
        except Exception as e:
            logger.error("Unexpected error (Groq): %s", e)
            return FALLBACK_GENERIC_ERROR


class HuggingFaceClient(LLMClient):
    """LLM client using HuggingFace Inference API.

    Uses the HuggingFace Router API (OpenAI-compatible) for chat completions.
    Model: meta-llama/Llama-3.2-3B-Instruct (free tier via serverless inference).
    Note: this model is gated — the HF account behind each API key must accept
    Meta's license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
    before it can be used, or requests will fail with a 403.
    """

    API_URL = "https://router.huggingface.co/v1/chat/completions"
    MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.MODEL

    def chat(self, messages: list[dict], params: Optional[InferenceParams] = None) -> str:
        """Send messages to HuggingFace Router API and return generated text."""
        params = params or inference_params
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "frequency_penalty": params.frequency_penalty,
                "presence_penalty": params.presence_penalty,
            }

            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                logger.warning("HuggingFace API rate limit exceeded")
                return FALLBACK_RATE_LIMIT

            if response.status_code >= 400:
                logger.error(
                    "HuggingFace API error: %d %s",
                    response.status_code,
                    response.text[:200],
                )
                return FALLBACK_GENERIC_ERROR

            data = response.json()
            # OpenAI-compatible chat completions format
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content:
                    return content

            logger.warning("HuggingFace API returned empty response")
            return FALLBACK_GENERIC_ERROR

        except requests.exceptions.Timeout as e:
            logger.error("Timeout error (HuggingFace): %s", e)
            return FALLBACK_CONNECTION_ERROR
        except requests.exceptions.ConnectionError as e:
            logger.error("Connection error (HuggingFace): %s", e)
            return FALLBACK_NETWORK_ERROR
        except (socket.gaierror, OSError) as e:
            logger.error("Network connectivity error (HuggingFace): %s", e)
            return FALLBACK_NETWORK_ERROR
        except requests.exceptions.RequestException as e:
            logger.error("Request error (HuggingFace): %s", e)
            return FALLBACK_GENERIC_ERROR
        except Exception as e:
            logger.error("Unexpected error (HuggingFace): %s", e)
            return FALLBACK_GENERIC_ERROR


class GeminiClient(LLMClient):
    """LLM client using Google Gemini free tier API.

    Uses the Gemini Pro model via the generativelanguage API.
    """

    DEFAULT_MODEL = "gemini-pro"
    API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def chat(self, messages: list[dict], params: Optional[InferenceParams] = None) -> str:
        """Send messages to Gemini API and return generated text."""
        params = params or inference_params
        try:
            url = f"{self.API_URL_BASE}/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}

            # Convert messages to Gemini format
            contents = self._convert_messages(messages)
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": params.temperature,
                    "topP": params.top_p,
                    "maxOutputTokens": params.max_tokens,
                },
            }

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                logger.warning("Gemini API rate limit exceeded")
                return FALLBACK_RATE_LIMIT

            if response.status_code >= 400:
                logger.error(
                    "Gemini API error: %d %s",
                    response.status_code,
                    response.text[:200],
                )
                return FALLBACK_GENERIC_ERROR

            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    text = parts[0].get("text", "")
                    if text:
                        return text

            logger.warning("Gemini API returned empty response")
            return FALLBACK_GENERIC_ERROR

        except requests.exceptions.Timeout as e:
            logger.error("Timeout error (Gemini): %s", e)
            return FALLBACK_CONNECTION_ERROR
        except requests.exceptions.ConnectionError as e:
            logger.error("Connection error (Gemini): %s", e)
            return FALLBACK_NETWORK_ERROR
        except (socket.gaierror, OSError) as e:
            logger.error("Network connectivity error (Gemini): %s", e)
            return FALLBACK_NETWORK_ERROR
        except requests.exceptions.RequestException as e:
            logger.error("Request error (Gemini): %s", e)
            return FALLBACK_GENERIC_ERROR
        except Exception as e:
            logger.error("Unexpected error (Gemini): %s", e)
            return FALLBACK_GENERIC_ERROR

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI-style messages to Gemini format.

        Gemini uses 'user' and 'model' roles with 'parts' containing text.
        System messages are prepended to the first user message.
        """
        contents = []
        system_text = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_text = content
                continue

            gemini_role = "model" if role == "assistant" else "user"

            # Prepend system text to first user message
            if system_text and gemini_role == "user":
                content = f"{system_text}\n\n{content}"
                system_text = ""

            contents.append(
                {"role": gemini_role, "parts": [{"text": content}]}
            )

        return contents


def create_llm_client(provider: str, api_key: str, model: Optional[str] = None) -> LLMClient:
    """Factory function to create the appropriate LLM client.

    Args:
        provider: Name of the LLM provider ("groq", "huggingface", "gemini", or "local").
        api_key: API key for the chosen provider (ignored for "local").
        model: Optional model name/path override. Defaults to the provider's
            built-in default model when omitted.

    Returns:
        An instance of the appropriate LLMClient subclass.

    Raises:
        ValueError: If the provider name is not recognized.
        NotImplementedError: If the provider is "local" — this build ships without
            a local inference backend. Wire your on-site local LLM client in here.
    """
    provider_lower = provider.lower().strip()

    if provider_lower == "groq":
        return GroqClient(api_key, model=model)
    elif provider_lower in ("huggingface", "hf"):
        return HuggingFaceClient(api_key, model=model)
    elif provider_lower == "gemini":
        return GeminiClient(api_key, model=model)
    elif provider_lower in ("local", "local_llm"):
        # Extension point: this build has no local inference backend wired in.
        raise NotImplementedError(
            "Provider 'local' is recognized but not implemented in this build. "
            "Plug in your on-site local LLM client here (must implement LLMClient.chat)."
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: groq, huggingface, gemini, local"
        )


class FailoverLLMClient(LLMClient):
    """LLM client that tries multiple providers in order.

    If the primary provider fails (rate limit, credits exhausted, error),
    automatically falls back to the next provider in the list.

    Attributes:
        clients: Ordered list of (provider_name, LLMClient) tuples.
        current_index: Index of the currently active provider.
    """

    # Responses that indicate we should try the next provider
    FAILOVER_INDICATORS = [
        FALLBACK_RATE_LIMIT,
        FALLBACK_GENERIC_ERROR,
        FALLBACK_CONNECTION_ERROR,
        FALLBACK_NETWORK_ERROR,
    ]

    def __init__(self, clients: list[tuple[str, "LLMClient"]]):
        """
        Args:
            clients: List of (provider_name, client_instance) tuples in priority order.
        """
        if not clients:
            raise ValueError("At least one LLM client must be provided")
        self.clients = clients
        self.current_index = 0

    def chat(self, messages: list[dict], params: Optional[InferenceParams] = None) -> str:
        """Try each provider in order until one succeeds.

        A "failure" is defined as returning one of the known fallback error messages.
        Post-processes the response to detect and truncate repetitive output.
        """
        last_response = ""

        for i in range(len(self.clients)):
            idx = (self.current_index + i) % len(self.clients)
            provider_name, client = self.clients[idx]

            response = client.chat(messages, params)

            # Check if this is a failure response
            if response in self.FAILOVER_INDICATORS:
                logger.warning(
                    "Provider '%s' failed, trying next... (%s)",
                    provider_name,
                    response[:60],
                )
                last_response = response
                continue

            # Success — update current index so next call starts here
            if idx != self.current_index:
                logger.info("Switched to provider '%s' after failover", provider_name)
                self.current_index = idx

            # Post-process: detect and truncate repetitive output
            response = self._truncate_repetition(response)

            return response

        # All providers failed — return the last error
        logger.error("All LLM providers failed")
        return last_response or FALLBACK_GENERIC_ERROR

    @staticmethod
    def _truncate_repetition(text: str, min_repeat_len: int = 10, max_repeats: int = 3) -> str:
        """Detect and truncate repetitive output from the LLM.

        Catches cases where the model starts repeating the same phrase/line
        over and over (common with small models hitting token limits).

        Strategy:
        1. Split into lines
        2. If the same line appears 3+ times consecutively, truncate
        3. If a phrase of 10+ chars repeats 3+ times, truncate
        """
        if not text or len(text) < 100:
            return text

        lines = text.split("\n")

        # Detect consecutive duplicate lines
        if len(lines) > 5:
            cleaned_lines = []
            repeat_count = 0
            last_line = None
            for line in lines:
                stripped = line.strip()
                if stripped == last_line and stripped:
                    repeat_count += 1
                    if repeat_count >= max_repeats:
                        continue  # Skip repeated lines
                else:
                    repeat_count = 0
                    last_line = stripped
                cleaned_lines.append(line)

            if len(cleaned_lines) < len(lines):
                text = "\n".join(cleaned_lines)
                logger.debug("Truncated %d repeated lines from LLM output", len(lines) - len(cleaned_lines))

        # Detect repeated phrases within the text (e.g., "- Poignard\n" repeated 30 times)
        import re
        # Find any phrase of 10+ chars that repeats 4+ times consecutively
        pattern = re.compile(r'(.{10,}?)\1{3,}', re.DOTALL)
        match = pattern.search(text)
        if match:
            # Keep only 2 occurrences of the repeated phrase
            repeated = match.group(1)
            replacement = repeated * 2 + "\n[... repeated content truncated ...]\n"
            text = text[:match.start()] + replacement + text[match.end():]
            logger.debug("Truncated repeated phrase pattern from LLM output")

        return text

    @property
    def active_provider(self) -> str:
        """Return the name of the currently active provider."""
        return self.clients[self.current_index][0]


def create_failover_client(config) -> LLMClient:
    """Create an LLM client with automatic failover between providers.

    For each provider in the failover order, creates a client for EACH
    available API key (e.g., HUGGINGFACE_API_KEY, HUGGINGFACE_API_KEY_2, etc.).
    Tries them all in order before moving to the next provider.

    Args:
        config: The application Config object.

    Returns:
        A FailoverLLMClient if multiple clients are configured,
        or a single LLMClient if only one is available.
    """
    clients = []

    # Map providers to their list of keys
    provider_keys = {
        "groq": config.groq_api_keys,
        "huggingface": config.huggingface_api_keys,
        "gemini": config.gemini_api_keys,
    }

    # Also include the primary key under its provider
    primary = config.llm_provider.lower().strip()
    if primary in provider_keys:
        # Ensure primary key is in the list (might already be via discovery)
        if config.llm_api_key not in provider_keys[primary]:
            provider_keys[primary].insert(0, config.llm_api_key)
    else:
        provider_keys[primary] = [config.llm_api_key]

    # Determine order
    if config.llm_failover_order:
        order = config.llm_failover_order
    else:
        # Default: primary first, then others
        order = [primary] + sorted(k for k in provider_keys if k != primary and provider_keys[k])

    # Create clients in order — for each provider, create one client per key
    for provider_name in order:
        keys = provider_keys.get(provider_name, [])
        for i, key in enumerate(keys):
            if not key:
                continue
            try:
                client = create_llm_client(provider_name, key)
                label = f"{provider_name}" if i == 0 else f"{provider_name}[key_{i+1}]"
                clients.append((label, client))
            except ValueError:
                logger.warning("Unknown provider in failover order: '%s'", provider_name)
                break  # Don't try more keys for unknown provider

    if not clients:
        return create_llm_client(primary, config.llm_api_key)

    if len(clients) == 1:
        return clients[0][1]

    logger.info(
        "LLM failover configured: %s",
        " → ".join(name for name, _ in clients),
    )
    return FailoverLLMClient(clients)
