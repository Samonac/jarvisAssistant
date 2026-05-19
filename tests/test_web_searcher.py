"""Tests for Web Searcher.

Property-based tests for result structure (Property 6) and
unit tests for error handling cases.
"""

from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.web_searcher import WebSearcher


# --- Strategies ---

# Strategy for generating valid search result dicts (as returned by duckduckgo-search)
valid_raw_result_strategy = st.fixed_dictionaries({
    "title": st.text(min_size=1, max_size=100, alphabet=st.characters(
        blacklist_categories=("Cs",), blacklist_characters=("\x00",)
    )).filter(lambda s: s.strip() != ""),
    "href": st.from_regex(r"https?://[a-z0-9]+\.[a-z]{2,6}(/[a-z0-9_-]*)*", fullmatch=True),
    "body": st.text(min_size=1, max_size=200, alphabet=st.characters(
        blacklist_categories=("Cs",), blacklist_characters=("\x00",)
    )).filter(lambda s: s.strip() != ""),
})

# Strategy for generating a list of valid raw results
valid_raw_results_list_strategy = st.lists(
    valid_raw_result_strategy, min_size=1, max_size=10
)

# Strategy for generating non-empty search queries
query_strategy = st.text(
    min_size=1, max_size=50,
    alphabet=st.characters(blacklist_categories=("Cs",))
).filter(lambda s: s.strip() != "")


class TestProperty6SearchResultsContainRequiredFields:
    """Property 6: Search results contain required fields.

    For any successful web search response, every result in the returned list
    SHALL contain non-empty `title`, `url`, and `snippet` fields.

    **Validates: Requirements 5.5**
    """

    @given(raw_results=valid_raw_results_list_strategy)
    @settings(max_examples=200)
    def test_all_results_have_required_fields(self, raw_results):
        """Every result returned by search has non-empty title, url, and snippet."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = raw_results
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            results = searcher.search("test query")

        # Every result must have all three required fields, non-empty
        for result in results:
            assert "title" in result, "Result missing 'title' field"
            assert "url" in result, "Result missing 'url' field"
            assert "snippet" in result, "Result missing 'snippet' field"
            assert isinstance(result["title"], str) and result["title"].strip() != ""
            assert isinstance(result["url"], str) and result["url"].strip() != ""
            assert isinstance(result["snippet"], str) and result["snippet"].strip() != ""

    @given(raw_results=valid_raw_results_list_strategy, query=query_strategy)
    @settings(max_examples=200)
    def test_result_fields_are_strings(self, raw_results, query):
        """All result fields are of type str."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = raw_results
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            results = searcher.search(query)

        for result in results:
            assert isinstance(result["title"], str)
            assert isinstance(result["url"], str)
            assert isinstance(result["snippet"], str)

    @given(raw_results=valid_raw_results_list_strategy)
    @settings(max_examples=100)
    def test_result_count_does_not_exceed_input(self, raw_results):
        """The number of formatted results never exceeds the raw input count."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = raw_results
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            results = searcher.search("test query")

        assert len(results) <= len(raw_results)

    @given(
        raw_results=st.lists(
            st.fixed_dictionaries({
                "title": st.just(""),
                "href": st.from_regex(r"https?://[a-z]+\.[a-z]{2,4}", fullmatch=True),
                "body": st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""),
            }),
            min_size=1, max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_results_with_empty_title_are_filtered(self, raw_results):
        """Results with empty title are excluded from output."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = raw_results
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            results = searcher.search("test query")

        # All results with empty title should be filtered out
        assert len(results) == 0


class TestErrorHandling:
    """Unit tests for error handling cases."""

    def test_empty_query_returns_empty_list(self):
        """An empty query string returns an empty list."""
        searcher = WebSearcher()
        result = searcher.search("")
        assert result == []

    def test_whitespace_only_query_returns_empty_list(self):
        """A whitespace-only query returns an empty list."""
        searcher = WebSearcher()
        result = searcher.search("   ")
        assert result == []

    def test_search_api_exception_returns_empty_list(self):
        """When the search API raises an exception, return an empty list."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.side_effect = Exception("Network error")
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = searcher.search("test query")

        assert result == []

    def test_search_timeout_returns_empty_list(self):
        """When the search times out, return an empty list."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.side_effect = TimeoutError("Request timed out")
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = searcher.search("test query")

        assert result == []

    def test_search_connection_error_returns_empty_list(self):
        """When there's a connection error, return an empty list."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.side_effect = ConnectionError("No internet")
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = searcher.search("test query")

        assert result == []

    def test_empty_results_returns_empty_list(self):
        """When the search returns no results, return an empty list."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = []
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = searcher.search("xyznonexistentquery123")

        assert result == []

    def test_none_results_returns_empty_list(self):
        """When the search returns None, return an empty list."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = None
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = searcher.search("test query")

        assert result == []

    def test_malformed_results_filtered_out(self):
        """Results missing required fields are filtered out."""
        searcher = WebSearcher()

        malformed_results = [
            {"title": "Good Title", "href": "https://example.com", "body": "Good snippet"},
            {"title": "", "href": "https://example.com", "body": "No title"},
            {"title": "No URL", "href": "", "body": "Missing URL"},
            {"title": "No Snippet", "href": "https://example.com", "body": ""},
            {},  # completely empty
        ]

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = malformed_results
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            results = searcher.search("test query")

        # Only the first result has all required fields
        assert len(results) == 1
        assert results[0]["title"] == "Good Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "Good snippet"

    def test_successful_search_returns_formatted_results(self):
        """A successful search returns properly formatted results."""
        searcher = WebSearcher()

        raw_results = [
            {
                "title": "Python Documentation",
                "href": "https://docs.python.org",
                "body": "Welcome to Python's official documentation.",
            },
            {
                "title": "Real Python",
                "href": "https://realpython.com",
                "body": "Python tutorials and articles.",
            },
        ]

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = raw_results
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            results = searcher.search("python")

        assert len(results) == 2
        assert results[0] == {
            "title": "Python Documentation",
            "url": "https://docs.python.org",
            "snippet": "Welcome to Python's official documentation.",
        }
        assert results[1] == {
            "title": "Real Python",
            "url": "https://realpython.com",
            "snippet": "Python tutorials and articles.",
        }

    def test_max_results_parameter_passed_to_ddgs(self):
        """The max_results parameter is passed to the DDGS text method."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = []
            mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
            mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs_class.return_value = mock_ddgs_instance

            searcher.search("test", max_results=3)

        mock_ddgs_instance.text.assert_called_once_with("test", max_results=3)

    def test_ddgs_constructor_exception_returns_empty_list(self):
        """When DDGS constructor raises, return an empty list."""
        searcher = WebSearcher()

        with patch("app.web_searcher.DDGS") as mock_ddgs_class:
            mock_ddgs_class.side_effect = RuntimeError("Failed to initialize")

            result = searcher.search("test query")

        assert result == []
