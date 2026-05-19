"""Web Searcher module.

Performs web searches using the DDGS (DuckDuckGo Search) library and returns
formatted results with title, URL, and snippet fields.
"""

import logging
import time

logger = logging.getLogger(__name__)


class WebSearcher:
    """Performs web searches using DuckDuckGo and returns structured results.

    Each result contains:
        - title (str): The page title
        - url (str): The page URL
        - snippet (str): A brief excerpt/description
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Perform a web search and return formatted results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (default: 5).

        Returns:
            A list of dicts with 'title', 'url', and 'snippet' keys.
            Returns an empty list if the search fails or returns no results.
        """
        if not query or not query.strip():
            return []

        # Try up to 2 times with a small delay between attempts
        for attempt in range(2):
            try:
                results = self._do_search(query, max_results)
                if results:
                    return results
                if attempt == 0:
                    logger.warning("Search returned empty for '%s', retrying...", query)
                    time.sleep(1)
            except Exception as e:
                logger.error("Search attempt %d failed for '%s': %s", attempt + 1, query, e)
                if attempt == 0:
                    time.sleep(1)

        logger.warning("All search attempts failed for query: '%s'", query)
        return []

    def _do_search(self, query: str, max_results: int) -> list[dict]:
        """Execute a single search attempt. Tries ddgs first, falls back to duckduckgo_search."""
        try:
            # Try the new 'ddgs' package first
            from ddgs import DDGS
            ddgs = DDGS()
            raw_results = list(ddgs.text(query, max_results=max_results))
        except ImportError:
            # Fall back to old package name
            from duckduckgo_search import DDGS
            ddgs = DDGS()
            raw_results = list(ddgs.text(query, max_results=max_results))

        logger.info("DuckDuckGo returned %d results for '%s'", len(raw_results), query[:50])

        if not raw_results:
            return []

        return self._format_results(raw_results)

    def _format_results(self, raw_results: list[dict]) -> list[dict]:
        """Extract and format title, URL, and snippet from raw search results."""
        formatted = []
        for result in raw_results:
            title = str(result.get("title", "")).strip()
            url = str(result.get("href", result.get("url", result.get("link", "")))).strip()
            snippet = str(result.get("body", result.get("snippet", result.get("description", "")))).strip()

            if title and url and snippet:
                formatted.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })

        return formatted
