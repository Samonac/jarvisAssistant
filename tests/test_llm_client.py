"""Property-based and unit tests for LLM Client module.

Tests Property 4 from the design document:
For any error response from the LLM API (connection errors, HTTP errors, timeouts),
the LLM Client SHALL return a non-empty fallback error message string rather than
raising an unhandled exception.
"""

import socket
from unittest.mock import patch, MagicMock

import pytest
import requests
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.llm_client import (
    LLMClient,
    GroqClient,
    HuggingFaceClient,
    GeminiClient,
    create_llm_client,
    FALLBACK_CONNECTION_ERROR,
    FALLBACK_RATE_LIMIT,
    FALLBACK_GENERIC_ERROR,
    FALLBACK_NETWORK_ERROR,
)


# --- Strategies ---

# Strategy for generating valid message lists
message_strategy = st.lists(
    st.fixed_dictionaries(
        {
            "role": st.sampled_from(["system", "user", "assistant"]),
            "content": st.text(min_size=1, max_size=100),
        }
    ),
    min_size=1,
    max_size=5,
)

# Strategy for generating HTTP error status codes
http_error_status_strategy = st.one_of(
    st.just(429),  # Rate limit
    st.integers(min_value=400, max_value=499),  # Client errors
    st.integers(min_value=500, max_value=599),  # Server errors
)

# Strategy for generating different exception types that can occur during requests
request_exception_strategy = st.sampled_from(
    [
        requests.exceptions.ConnectionError("Connection refused"),
        requests.exceptions.Timeout("Request timed out"),
        requests.exceptions.RequestException("Generic request error"),
        socket.gaierror("Name resolution failed"),
        OSError("Network unreachable"),
    ]
)

# Strategy for selecting a client class
client_class_strategy = st.sampled_from(
    [GroqClient, HuggingFaceClient, GeminiClient]
)


# --- Property Tests ---


class TestProperty4LLMErrorsFallback:
    """Property 4: LLM errors produce fallback message.

    For any error response from the LLM API (connection errors, HTTP errors,
    timeouts), the LLM Client SHALL return a non-empty fallback error message
    string rather than raising an unhandled exception.

    **Validates: Requirements 2.3**
    """

    @given(
        messages=message_strategy,
        status_code=http_error_status_strategy,
    )
    @settings(max_examples=100)
    def test_groq_http_errors_return_fallback(self, messages, status_code):
        """GroqClient returns a non-empty fallback string for any HTTP error."""
        client = GroqClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = "Error response body"

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat(messages)

        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        messages=message_strategy,
        status_code=http_error_status_strategy,
    )
    @settings(max_examples=100)
    def test_huggingface_http_errors_return_fallback(self, messages, status_code):
        """HuggingFaceClient returns a non-empty fallback string for any HTTP error."""
        client = HuggingFaceClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = "Error response body"

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat(messages)

        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        messages=message_strategy,
        status_code=http_error_status_strategy,
    )
    @settings(max_examples=100)
    def test_gemini_http_errors_return_fallback(self, messages, status_code):
        """GeminiClient returns a non-empty fallback string for any HTTP error."""
        client = GeminiClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = "Error response body"

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat(messages)

        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        messages=message_strategy,
        exception=request_exception_strategy,
    )
    @settings(max_examples=100)
    def test_groq_connection_errors_return_fallback(self, messages, exception):
        """GroqClient returns a non-empty fallback string for any connection error."""
        client = GroqClient(api_key="test-key")

        with patch("app.llm_client.requests.post", side_effect=exception):
            result = client.chat(messages)

        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        messages=message_strategy,
        exception=request_exception_strategy,
    )
    @settings(max_examples=100)
    def test_huggingface_connection_errors_return_fallback(self, messages, exception):
        """HuggingFaceClient returns a non-empty fallback string for any connection error."""
        client = HuggingFaceClient(api_key="test-key")

        with patch("app.llm_client.requests.post", side_effect=exception):
            result = client.chat(messages)

        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        messages=message_strategy,
        exception=request_exception_strategy,
    )
    @settings(max_examples=100)
    def test_gemini_connection_errors_return_fallback(self, messages, exception):
        """GeminiClient returns a non-empty fallback string for any connection error."""
        client = GeminiClient(api_key="test-key")

        with patch("app.llm_client.requests.post", side_effect=exception):
            result = client.chat(messages)

        assert isinstance(result, str)
        assert len(result) > 0


# --- Unit Tests ---


class TestGroqClient:
    """Unit tests for GroqClient."""

    def test_rate_limit_returns_specific_message(self):
        """HTTP 429 returns the rate limit fallback message."""
        client = GroqClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_RATE_LIMIT

    def test_timeout_returns_connection_error_message(self):
        """Timeout returns the connection disrupted fallback message."""
        client = GroqClient(api_key="test-key")

        with patch(
            "app.llm_client.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_CONNECTION_ERROR

    def test_connection_error_returns_network_message(self):
        """ConnectionError returns the network connectivity fallback message."""
        client = GroqClient(api_key="test-key")

        with patch(
            "app.llm_client.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_NETWORK_ERROR

    def test_socket_error_returns_network_message(self):
        """Socket gaierror returns the network connectivity fallback message."""
        client = GroqClient(api_key="test-key")

        with patch(
            "app.llm_client.requests.post",
            side_effect=socket.gaierror("Name resolution failed"),
        ):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_NETWORK_ERROR

    def test_server_error_returns_generic_message(self):
        """HTTP 500 returns the generic fallback message."""
        client = GroqClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_GENERIC_ERROR

    def test_successful_streaming_response(self):
        """Successful streaming response returns accumulated content."""
        client = GroqClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" Sir"}}]}',
            "data: [DONE]",
        ]

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == "Hello Sir"

    def test_empty_streaming_response_returns_fallback(self):
        """Empty streaming response returns generic fallback."""
        client = GroqClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = ["data: [DONE]"]

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_GENERIC_ERROR


class TestHuggingFaceClient:
    """Unit tests for HuggingFaceClient."""

    def test_rate_limit_returns_specific_message(self):
        """HTTP 429 returns the rate limit fallback message."""
        client = HuggingFaceClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_RATE_LIMIT

    def test_timeout_returns_connection_error_message(self):
        """Timeout returns the connection disrupted fallback message."""
        client = HuggingFaceClient(api_key="test-key")

        with patch(
            "app.llm_client.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_CONNECTION_ERROR

    def test_successful_response(self):
        """Successful response returns the generated content."""
        client = HuggingFaceClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello Sir, how may I assist?"}}]
        }

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == "Hello Sir, how may I assist?"

    def test_empty_response_returns_fallback(self):
        """Empty response body returns generic fallback."""
        client = HuggingFaceClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_GENERIC_ERROR


class TestGeminiClient:
    """Unit tests for GeminiClient."""

    def test_rate_limit_returns_specific_message(self):
        """HTTP 429 returns the rate limit fallback message."""
        client = GeminiClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_RATE_LIMIT

    def test_timeout_returns_connection_error_message(self):
        """Timeout returns the connection disrupted fallback message."""
        client = GeminiClient(api_key="test-key")

        with patch(
            "app.llm_client.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_CONNECTION_ERROR

    def test_successful_response(self):
        """Successful response returns the generated content."""
        client = GeminiClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Greetings, Sir."}]}}]
        }

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == "Greetings, Sir."

    def test_empty_response_returns_fallback(self):
        """Empty response body returns generic fallback."""
        client = GeminiClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}

        with patch("app.llm_client.requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])

        assert result == FALLBACK_GENERIC_ERROR

    def test_message_conversion_system_prompt(self):
        """System messages are prepended to the first user message in Gemini format."""
        client = GeminiClient(api_key="test-key")
        messages = [
            {"role": "system", "content": "You are Jarvis."},
            {"role": "user", "content": "Hello"},
        ]

        contents = client._convert_messages(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert "You are Jarvis." in contents[0]["parts"][0]["text"]
        assert "Hello" in contents[0]["parts"][0]["text"]

    def test_message_conversion_roles(self):
        """User and assistant roles are correctly mapped."""
        client = GeminiClient(api_key="test-key")
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello Sir"},
            {"role": "user", "content": "How are you?"},
        ]

        contents = client._convert_messages(messages)

        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[2]["role"] == "user"


class TestCreateLLMClient:
    """Unit tests for the create_llm_client factory function."""

    def test_creates_groq_client(self):
        """Provider 'groq' creates a GroqClient."""
        client = create_llm_client("groq", "test-key")
        assert isinstance(client, GroqClient)

    def test_creates_huggingface_client(self):
        """Provider 'huggingface' creates a HuggingFaceClient."""
        client = create_llm_client("huggingface", "test-key")
        assert isinstance(client, HuggingFaceClient)

    def test_creates_huggingface_client_short_name(self):
        """Provider 'hf' creates a HuggingFaceClient."""
        client = create_llm_client("hf", "test-key")
        assert isinstance(client, HuggingFaceClient)

    def test_creates_gemini_client(self):
        """Provider 'gemini' creates a GeminiClient."""
        client = create_llm_client("gemini", "test-key")
        assert isinstance(client, GeminiClient)

    def test_case_insensitive(self):
        """Provider name matching is case-insensitive."""
        assert isinstance(create_llm_client("GROQ", "key"), GroqClient)
        assert isinstance(create_llm_client("Gemini", "key"), GeminiClient)
        assert isinstance(create_llm_client("HuggingFace", "key"), HuggingFaceClient)

    def test_unknown_provider_raises_value_error(self):
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client("unknown_provider", "test-key")

    def test_empty_provider_raises_value_error(self):
        """Empty provider string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client("", "test-key")


class TestLLMClientAbstract:
    """Unit tests for the abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """LLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMClient()
