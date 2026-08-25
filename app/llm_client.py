"""LLM Client module for Jarvis Assistant.

Rewired to talk to the local LLM Gateway (FastAPI) instead of calling
cloud providers directly.  The Gateway lives at LLM_GATEWAY_URL
(default: http://llm-server.local:8000) and exposes:

    POST /chat          – blocking JSON response
    POST /chat/stream   – SSE token stream

Profile selection is automatic: the client inspects the message list
and picks the right profile so the gateway can route to the correct
local model without the Pi ever knowing which model is behind it.

Profiles
--------
  fast      → qwen3:8b         (temperature 0.2, think=False)
  coding    → qwen2.5-coder:14b (temperature 0.1, think=False)
  reasoning → deepseek-r1:32b   (temperature 0.5, think=True)

Public API (unchanged from previous version)
--------------------------------------------
  llm_client.chat(messages: list[dict]) -> str
  create_failover_client(config) -> LLMClient
  inference_params  (global, modified from the UI)
"""

import json
import logging
import re
import socket
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback messages (kept verbatim for compatibility with FailoverLLMClient)
# ---------------------------------------------------------------------------

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

# Default timeout — gateway handles the per-profile timeout internally.
# Client side: generous for reasoning (32b can take 5+ min), tight for fast.
REQUEST_TIMEOUT = 120
REQUEST_TIMEOUT_REASONING = 660  # 11 minutes — covers 32b worst case


# ---------------------------------------------------------------------------
# InferenceParams — still exposed so the UI can tweak values at runtime.
# When a per-request override is set the gateway will honour it over the
# profile default (via the temperature / top_p override fields).
# ---------------------------------------------------------------------------

class InferenceParams:
    """Configurable inference parameters.

    Attributes:
        temperature: Controls randomness (0.0–2.0). Default 0.7.
        top_p: Nucleus sampling threshold. Default 0.9.
        max_tokens: Maximum tokens to generate. Default 1024.
        frequency_penalty: Penalise repeated tokens. Default 0.0.
        presence_penalty: Penalise new topics. Default 0.0.
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
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


# Global instance — modified at runtime via the UI
inference_params = InferenceParams()


# ---------------------------------------------------------------------------
# Profile selector — automatic reasoning-level detection
# ---------------------------------------------------------------------------

# Keywords that clearly signal multi-step reasoning is required
_REASONING_PATTERNS = re.compile(
    r"""
    \b(
        analy[sz]e | compar(e|ison) | evaluat(e|ion) | asses(s|ment)
      | explain\s+why | explain\s+how | reason(ing)? | infer(ence)?
      | deduc(e|tion) | hypothes[ie]s | diagnos(e|is) | design\s+a
      | architect(ure)? | plan\s+(the|a|an) | strateg(y|ize)
      | proof | prove | theorem | mathematical | algebraic
      | step[\s\-]by[\s\-]step | think\s+through | walk\s+me\s+through
      | pros\s+and\s+cons | trade[\s\-]?off
      | what\s+would\s+happen | what\s+if | scenario
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Keywords that signal a coding task
_CODING_PATTERNS = re.compile(
    r"""
    \b(
        code | coding | script | function | class | method | module
      | implement | refactor | debug | fix\s+(the\s+)?(bug|error|issue|code)
      | write\s+(a\s+)?(script|function|class|code|program|test)
      | unit\s+test | pytest | unittest | algorithm | snippet
      | python | javascript | typescript | bash | shell | sql | yaml | json
      | api | endpoint | regex | parse | compile | syntax
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


# Simple-query detector — mirrors the gateway heuristic
_SIMPLE_QUERY_RE = re.compile(
    r"""
    ^(
        what\s+is\s+[\d\s\+\-\*\/\^\(\)\.]+\??$
      | [\d\s\+\-\*\/\^\(\)\.]+\=\?$
      | (what|how\s+much|convert)\s+.{0,60}\?$
      | (what\s+(time|date|day)|current\s+(time|date))\b
      | (hi|hello|hey|good\s+(morning|afternoon|evening))\b
      | (thanks?|thank\s+you)\b
      | (yes|no|ok(ay)?|sure|got\s+it)\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Phrases that turn thinking mode ON for the session
_THINK_ON_RE = re.compile(
    r"\b(start\s+think(ing)?|think\s+(deeply|harder|more)|use\s+(reasoning|deepseek)|"
    r"reason\s+through|take\s+your\s+time|think\s+about\s+this\s+carefully|"
    r"use\s+more\s+tokens?|reasoning\s+mode|deep\s+think)\b",
    re.IGNORECASE,
)

# Phrases that turn thinking mode OFF for the session
_THINK_OFF_RE = re.compile(
    r"\b(stop\s+think(ing)?|no\s+(more\s+)?think(ing)?|answer\s+quick(ly)?|"
    r"be\s+brief|keep\s+it\s+short|fast\s+mode|quick\s+(answer|reply|response)|"
    r"don'?t\s+(over)?think|stop\s+over.?think(ing)?|less\s+think(ing)?)\b",
    re.IGNORECASE,
)

# Condensed personality stub used instead of the full JARVIS system prompt
# for simple queries — ~30 tokens vs ~400, cuts prefill time significantly
_SIMPLE_SYSTEM_STUB = (
    "You are J.A.R.V.I.S., a witty British AI assistant. "
    "Address the user as 'Sir'. Be brief and precise."
)


def _check_simple_query(messages: list[dict]) -> bool:
    """Return True if the last user message is a trivial single-fact query."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return False
    text = user_msgs[-1].get("content", "").strip()
    return len(text) < 80 and bool(_SIMPLE_QUERY_RE.match(text))


def _select_profile(messages: list[dict]) -> str:
    """Inspect the conversation and return the best profile name.

    Decision logic (in priority order):
      1. reasoning  — last user message contains deep-analysis keywords
                      OR conversation is long (>8 turns, context required)
      2. coding     — last user message contains code/programming keywords
                      OR any message contains a code block (```…```)
      3. fast       — everything else

    Returns one of: "fast", "coding", "reasoning"
    """
    # Gather the last user message and full text for context
    last_user = ""
    full_text = ""
    for m in messages:
        content = m.get("content", "")
        full_text += content + " "
        if m.get("role") == "user":
            last_user = content

    # Conversation depth (number of turns)
    turn_count = sum(1 for m in messages if m.get("role") == "user")

    # --- Reasoning triggers ---
    if _REASONING_PATTERNS.search(last_user):
        logger.debug("Profile selected: reasoning (keyword match in user message)")
        return "reasoning"

    # Long multi-turn conversation → benefit from deeper model
    if turn_count > 8:
        logger.debug("Profile selected: reasoning (long conversation, %d turns)", turn_count)
        return "reasoning"

    # --- Coding triggers ---
    if _CODING_PATTERNS.search(last_user):
        logger.debug("Profile selected: coding (keyword match in user message)")
        return "coding"

    # Code block present anywhere in conversation
    if "```" in full_text:
        logger.debug("Profile selected: coding (code block detected)")
        return "coding"

    # --- Default ---
    logger.debug("Profile selected: fast (no special signals)")
    return "fast"


# ---------------------------------------------------------------------------
# Abstract base (unchanged public contract)
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base class for LLM provider clients."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        params: Optional["InferenceParams"] = None,
    ) -> str:
        """Send messages and return generated text.

        Args:
            messages: List of dicts with 'role' and 'content' keys.

        Returns:
            Generated text string.  On error returns a FALLBACK_* constant.
        """


# ---------------------------------------------------------------------------
# GatewayClient — the new primary implementation
# ---------------------------------------------------------------------------

class GatewayClient(LLMClient):
    """Talks to the local LLM Gateway (FastAPI on the Mac).

    Automatically selects a reasoning profile based on the conversation
    content.  Uses the blocking /chat endpoint; streaming is available
    via stream_chat() for callers that want it.

    Args:
        base_url: Gateway base URL, e.g. "http://llm-server.local:8000".
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._chat_url = f"{self.base_url}/chat"
        self._stream_url = f"{self.base_url}/chat/stream"
        # Session-level think mode — toggled by user phrases like
        # "start thinking" / "stop thinking".  When True, every subsequent
        # call uses the reasoning profile regardless of content heuristics.
        self._think_mode: bool = False

    # ------------------------------------------------------------------
    # Blocking chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        params: Optional[InferenceParams] = None,
    ) -> str:
        """Send messages to the gateway and return the assistant reply."""
        # Check last user message for session-level think-mode toggles
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            last_user_text = user_msgs[-1].get("content", "")
            if _THINK_ON_RE.search(last_user_text):
                self._think_mode = True
                logger.info("Think mode ENABLED for session")
            elif _THINK_OFF_RE.search(last_user_text):
                self._think_mode = False
                logger.info("Think mode DISABLED for session")

        # Profile: session think_mode overrides content heuristics
        if self._think_mode:
            profile = "reasoning"
        else:
            profile = _select_profile(messages)

        simple = (not self._think_mode) and _check_simple_query(messages)
        logger.info(
            "Gateway request — profile: %s%s%s",
            profile,
            " [think_mode]" if self._think_mode else "",
            " [simple]" if simple else "",
        )

        # Separate system messages from the conversation
        system_content = None
        conv_messages = []
        for m in messages:
            if m.get("role") == "system":
                # Concatenate multiple system messages
                system_content = (
                    (system_content + "\n\n" + m["content"])
                    if system_content
                    else m["content"]
                )
            else:
                conv_messages.append({"role": m["role"], "content": m["content"]})

        # For simple queries replace the full ~400-token JARVIS system prompt
        # with a 30-token stub — dramatically cuts prefill time
        if simple and system_content:
            system_content = _SIMPLE_SYSTEM_STUB

        params = params or inference_params
        payload: dict = {
            "profile": profile,
            "messages": conv_messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "simple_query": simple,
        }
        if system_content:
            payload["system"] = system_content

        try:
            timeout = REQUEST_TIMEOUT_REASONING if profile == "reasoning" else REQUEST_TIMEOUT
            resp = requests.post(
                self._chat_url,
                json=payload,
                timeout=timeout,
            )

            if resp.status_code == 429:
                logger.warning("Gateway rate limit")
                return FALLBACK_RATE_LIMIT

            if resp.status_code >= 400:
                logger.error("Gateway error: %d %s", resp.status_code, resp.text[:200])
                return FALLBACK_GENERIC_ERROR

            data = resp.json()
            content = data.get("message", {}).get("content", "")
            if not content:
                logger.warning("Gateway returned empty content")
                return FALLBACK_GENERIC_ERROR

            logger.info(
                "Gateway response — model: %s  prompt: %d  completion: %d  total: %d  %.0f ms",
                data.get("model", "?"),
                data.get("prompt_tokens", 0),
                data.get("completion_tokens", 0),
                data.get("total_tokens", 0),
                data.get("latency_ms", 0),
            )
            return content

        except requests.exceptions.Timeout:
            logger.error("Timeout waiting for gateway (%s)", self._chat_url)
            return FALLBACK_CONNECTION_ERROR
        except requests.exceptions.ConnectionError:
            logger.error("Cannot reach gateway at %s", self.base_url)
            return FALLBACK_NETWORK_ERROR
        except (socket.gaierror, OSError) as exc:
            logger.error("Network error reaching gateway: %s", exc)
            return FALLBACK_NETWORK_ERROR
        except Exception as exc:
            logger.error("Unexpected gateway error: %s", exc)
            return FALLBACK_GENERIC_ERROR

    # ------------------------------------------------------------------
    # Internal utility chat — for title generation, summaries, etc.
    # Hard-capped at 64 tokens, no system prompt overhead, no thinking.
    # ------------------------------------------------------------------

    def quick_chat(self, prompt: str) -> str:
        """Minimal single-turn call for internal utility tasks.

        Bypasses profile selection, history, and system prompt.
        Uses the fast model with a 64-token output cap and simple_query=True
        so Qwen3 skips its internal CoT entirely.
        """
        payload: dict = {
            "profile": "fast",
            "messages": [{"role": "user", "content": prompt}],
            "simple_query": True,
        }
        try:
            resp = requests.post(
                self._chat_url,
                json=payload,
                timeout=30,
            )
            if resp.status_code >= 400:
                return ""
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as exc:
            logger.debug("quick_chat failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Streaming helper (optional, for callers that want token-by-token)
    # ------------------------------------------------------------------

    def stream_chat(self, messages: list[dict]):
        """Generator that yields text tokens as they arrive via SSE.

        Usage::
            for token in client.stream_chat(messages):
                ui.append(token)
        """
        profile = _select_profile(messages)

        system_content = None
        conv_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_content = (
                    (system_content + "\n\n" + m["content"])
                    if system_content
                    else m["content"]
                )
            else:
                conv_messages.append({"role": m["role"], "content": m["content"]})

        payload: dict = {
            "profile": profile,
            "messages": conv_messages,
            "temperature": inference_params.temperature,
            "top_p": inference_params.top_p,
        }
        if system_content:
            payload["system"] = system_content

        try:
            resp = requests.post(
                self._stream_url,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                if data.startswith("[ERROR]"):
                    logger.error("Gateway stream error: %s", data)
                    break
                yield data

        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield FALLBACK_GENERIC_ERROR


# ---------------------------------------------------------------------------
# Legacy cloud clients — kept so existing .env configs with LLM_PROVIDER=groq
# etc. don't break.  They are used only when the gateway is unreachable.
# ---------------------------------------------------------------------------

class GroqClient(LLMClient):
    """Fallback: Groq cloud API (OpenAI-compatible streaming)."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.MODEL

    def chat(
        self,
        messages: list[dict],
        params: Optional[InferenceParams] = None,
    ) -> str:
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
                self.API_URL, headers=headers, json=payload,
                stream=True, timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429:
                return FALLBACK_RATE_LIMIT
            if response.status_code >= 400:
                logger.error("Groq error: %d", response.status_code)
                return FALLBACK_GENERIC_ERROR

            content = []
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        content.append(delta["content"])
                except (ValueError, KeyError, IndexError):
                    continue

            return "".join(content) or FALLBACK_GENERIC_ERROR

        except requests.exceptions.Timeout:
            return FALLBACK_CONNECTION_ERROR
        except requests.exceptions.ConnectionError:
            return FALLBACK_NETWORK_ERROR
        except (socket.gaierror, OSError):
            return FALLBACK_NETWORK_ERROR
        except Exception as exc:
            logger.error("Groq unexpected error: %s", exc)
            return FALLBACK_GENERIC_ERROR


class HuggingFaceClient(LLMClient):
    """Fallback: HuggingFace / local Ollama OpenAI-compat endpoint."""

    API_URL = "http://192.168.1.53:11434/v1/chat/completions"
    MODEL = "qwen3:8b"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.MODEL

    def chat(
        self,
        messages: list[dict],
        params: Optional[InferenceParams] = None,
    ) -> str:
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
            }
            response = requests.post(
                self.API_URL, headers=headers, json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429:
                return FALLBACK_RATE_LIMIT
            if response.status_code >= 400:
                logger.error("HuggingFace error: %d", response.status_code)
                return FALLBACK_GENERIC_ERROR

            data = response.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content:
                    return content

            return FALLBACK_GENERIC_ERROR

        except requests.exceptions.Timeout:
            return FALLBACK_CONNECTION_ERROR
        except requests.exceptions.ConnectionError:
            return FALLBACK_NETWORK_ERROR
        except Exception as exc:
            logger.error("HuggingFace unexpected error: %s", exc)
            return FALLBACK_GENERIC_ERROR


class GeminiClient(LLMClient):
    """Fallback: Google Gemini free-tier API."""

    API_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-pro:generateContent"
    )

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or "gemini-pro"

    def chat(
        self,
        messages: list[dict],
        params: Optional[InferenceParams] = None,
    ) -> str:
        params = params or inference_params
        try:
            url = self.API_URL_TEMPLATE.replace("gemini-pro", self.model)
            url = f"{url}?key={self.api_key}"
            payload = {
                "contents": self._convert_messages(messages),
                "generationConfig": {
                    "temperature": params.temperature,
                    "topP": params.top_p,
                    "maxOutputTokens": params.max_tokens,
                },
            }
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                return FALLBACK_RATE_LIMIT
            if response.status_code >= 400:
                logger.error("Gemini error: %d", response.status_code)
                return FALLBACK_GENERIC_ERROR

            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "") or FALLBACK_GENERIC_ERROR

            return FALLBACK_GENERIC_ERROR

        except requests.exceptions.Timeout:
            return FALLBACK_CONNECTION_ERROR
        except requests.exceptions.ConnectionError:
            return FALLBACK_NETWORK_ERROR
        except Exception as exc:
            logger.error("Gemini unexpected error: %s", exc)
            return FALLBACK_GENERIC_ERROR

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        contents = []
        system_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
                continue
            gemini_role = "model" if role == "assistant" else "user"
            if system_text and gemini_role == "user":
                content = f"{system_text}\n\n{content}"
                system_text = ""
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        return contents


# ---------------------------------------------------------------------------
# FailoverLLMClient — wraps GatewayClient + legacy cloud clients
# ---------------------------------------------------------------------------

class FailoverLLMClient(LLMClient):
    """Tries clients in order until one succeeds.

    Attributes:
        clients: Ordered list of (label, LLMClient) tuples.
        current_index: Index of the currently active client.
    """

    FAILOVER_INDICATORS = [
        FALLBACK_RATE_LIMIT,
        FALLBACK_GENERIC_ERROR,
        FALLBACK_CONNECTION_ERROR,
        FALLBACK_NETWORK_ERROR,
    ]

    def __init__(self, clients: list[tuple[str, "LLMClient"]]):
        if not clients:
            raise ValueError("At least one LLM client must be provided")
        self.clients = clients
        self.current_index = 0

    def chat(
        self,
        messages: list[dict],
        params: Optional[InferenceParams] = None,
    ) -> str:
        last_response = ""
        for i in range(len(self.clients)):
            idx = (self.current_index + i) % len(self.clients)
            label, client = self.clients[idx]
            response = client.chat(messages, params)
            if response in self.FAILOVER_INDICATORS:
                logger.warning("Client '%s' failed (%s…), trying next", label, response[:60])
                last_response = response
                continue
            if idx != self.current_index:
                logger.info("Switched to client '%s' after failover", label)
                self.current_index = idx
            return self._truncate_repetition(response)
        logger.error("All LLM clients failed")
        return last_response or FALLBACK_GENERIC_ERROR

    @staticmethod
    def _truncate_repetition(text: str, max_repeats: int = 3) -> str:
        if not text or len(text) < 100:
            return text
        lines = text.split("\n")
        if len(lines) > 5:
            cleaned, repeat_count, last_line = [], 0, None
            for line in lines:
                stripped = line.strip()
                if stripped == last_line and stripped:
                    repeat_count += 1
                    if repeat_count >= max_repeats:
                        continue
                else:
                    repeat_count, last_line = 0, stripped
                cleaned.append(line)
            text = "\n".join(cleaned)
        pattern = re.compile(r'(.{10,}?)\1{3,}', re.DOTALL)
        match = pattern.search(text)
        if match:
            repeated = match.group(1)
            text = (
                text[:match.start()]
                + repeated * 2
                + "\n[... repeated content truncated ...]\n"
                + text[match.end():]
            )
        return text

    @property
    def active_provider(self) -> str:
        return self.clients[self.current_index][0]

    def quick_chat(self, prompt: str) -> str:
        """Run a utility request through the active client when supported."""
        for _, client in self.clients:
            quick_chat = getattr(client, "quick_chat", None)
            if quick_chat:
                response = quick_chat(prompt)
                if response:
                    return response
        return ""


# ---------------------------------------------------------------------------
# Factory functions (unchanged public signatures)
# ---------------------------------------------------------------------------

def create_llm_client(
    provider: str,
    api_key: str,
    model: Optional[str] = None,
) -> LLMClient:
    """Create a single LLM client by provider name.

    "gateway" is the new preferred provider; legacy names still work.
    """
    provider_lower = provider.lower().strip()
    if provider_lower == "gateway":
        return GatewayClient(api_key)  # api_key is the base URL here
    elif provider_lower in ("local", "local_llm"):
        raise NotImplementedError(
            "Provider 'local' is recognized but not implemented in this build."
        )
    elif provider_lower == "groq":
        return GroqClient(api_key, model=model)
    elif provider_lower in ("huggingface", "hf"):
        return HuggingFaceClient(api_key, model=model)
    elif provider_lower == "gemini":
        return GeminiClient(api_key, model=model)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: gateway, groq, huggingface, gemini"
        )


def create_failover_client(config) -> LLMClient:
    """Create an LLM client, preferring the local gateway with cloud fallback.

    Resolution order
    ----------------
    1. If LLM_GATEWAY_URL is set (or the default host resolves), use
       GatewayClient as the primary client.
    2. Then add legacy cloud clients from LLM_FAILOVER_ORDER as fallbacks.
    3. If only one client ends up in the list, return it directly
       (avoids unnecessary FailoverLLMClient wrapper).

    The gateway URL defaults to http://llm-server.local:8000 but can be
    overridden with LLM_GATEWAY_URL in .env.
    """
    import os
    clients: list[tuple[str, LLMClient]] = []

    # --- Primary: local gateway ---
    gateway_url = os.environ.get("LLM_GATEWAY_URL", "http://192.168.1.52:8000")
    clients.append(("gateway", GatewayClient(gateway_url)))
    logger.info("LLM Gateway configured at %s", gateway_url)

    # --- Fallbacks: legacy cloud providers ---
    provider_keys: dict[str, list[str]] = {
        "groq": config.groq_api_keys,
        "huggingface": config.huggingface_api_keys,
        "gemini": config.gemini_api_keys,
    }

    primary = config.llm_provider.lower().strip()
    if primary not in ("gateway",):
        if primary in provider_keys:
            if config.llm_api_key not in provider_keys[primary]:
                provider_keys[primary].insert(0, config.llm_api_key)
        else:
            provider_keys[primary] = [config.llm_api_key]

    order = config.llm_failover_order or [
        primary, *sorted(k for k in provider_keys if k != primary and provider_keys[k])
    ]

    for provider_name in order:
        if provider_name == "gateway":
            continue  # already added
        keys = provider_keys.get(provider_name, [])
        for i, key in enumerate(keys):
            if not key:
                continue
            try:
                client = create_llm_client(provider_name, key)
                label = provider_name if i == 0 else f"{provider_name}[key_{i + 1}]"
                clients.append((label, client))
            except ValueError:
                logger.warning("Unknown provider in failover order: '%s'", provider_name)
                break

    if len(clients) == 1:
        return clients[0][1]

    logger.info(
        "LLM failover chain: %s",
        " → ".join(label for label, _ in clients),
    )
    return FailoverLLMClient(clients)
