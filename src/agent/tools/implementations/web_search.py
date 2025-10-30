"""
Web search tool implementation using DuckDuckGo.
"""

from typing import Any
from duckduckgo_search import DDGS
from ..base import SearchTool, ToolMetadata, ToolValidationException


class WebSearchTool(SearchTool):
    """
    Web search tool for finding defect-related information.

    Uses DuckDuckGo for privacy-focused searches.
    """

    def __init__(self, max_results: int = 5):
        super().__init__()
        self.max_results = max_results

    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name="web_search",
            description="Search the web for software defect information, bug patterns, and best practices",
            parameters={
                "query": "Search query string (required)",
                "max_results": f"Maximum results to return (default: {self.max_results})"
            },
            returns="Formatted search results with titles, snippets, and links",
            examples=[
                "query='Python list index out of range error'",
                "query='null pointer dereference C++'"
            ]
        )

    def validate_parameters(self, **kwargs) -> bool:
        """Validate search parameters."""
        query = kwargs.get("query", "")
        if not query or not isinstance(query, str):
            raise ToolValidationException("Query must be a non-empty string")

        if len(query) < 3:
            raise ToolValidationException("Query must be at least 3 characters")

        return True

    def search(self, query: str, **kwargs) -> str:
        """
        Perform web search.

        Args:
            query: Search query
            **kwargs: Additional parameters (max_results)

        Returns:
            Formatted search results
        """
        # Optimize query for defect prediction
        optimized_query = self._optimize_query(query)
        max_results = kwargs.get("max_results", self.max_results)

        try:
            # Perform search
            with DDGS() as ddgs:
                results = list(ddgs.text(optimized_query, max_results=max_results))

            # Format results
            return self._format_results(results)

        except Exception as e:
            return f"Search failed: {str(e)}"

    def _optimize_query(self, query: str) -> str:
        """
        Optimize search query for defect-related information.

        Args:
            query: Original query

        Returns:
            Optimized query
        """
        # Add defect-related keywords if not present
        defect_keywords = ["software", "defect", "bug", "error", "analysis"]

        # Check if query already contains defect-related terms
        query_lower = query.lower()
        has_defect_term = any(kw in query_lower for kw in defect_keywords)

        if not has_defect_term:
            query = f"{query} software defect bug analysis"

        return query

    def _format_results(self, results: list) -> str:
        """
        Format search results as readable text.

        Args:
            results: Raw search results from DuckDuckGo

        Returns:
            Formatted string
        """
        if not results:
            return "No search results found."

        formatted_parts = []
        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            snippet = result.get("body", "No description")
            link = result.get("href", "No link")

            formatted_parts.append(
                f"{idx}. {title}\n"
                f"   {snippet}\n"
                f"   Source: {link}"
            )

        return "\n\n".join(formatted_parts)
