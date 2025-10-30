"""
Base tool interface for the defect prediction agent.
All tools must implement this protocol.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToolMetadata:
    """Metadata describing a tool's capabilities."""
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: str
    examples: Optional[list] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples or []
        }


class ToolException(Exception):
    """Base exception for tool-related errors."""
    pass


class ToolExecutionException(ToolException):
    """Raised when tool execution fails."""
    pass


class ToolValidationException(ToolException):
    """Raised when tool input validation fails."""
    pass


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Tools are discrete capabilities that the agent can invoke
    to gather information or perform actions.
    """

    def __init__(self):
        self._metadata: Optional[ToolMetadata] = None
        self._is_initialized = False

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result

        Raises:
            ToolExecutionException: If execution fails
            ToolValidationException: If parameters are invalid
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """
        Get tool metadata describing its capabilities.

        Returns:
            ToolMetadata object
        """
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate input parameters.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid, raises exception otherwise

        Raises:
            ToolValidationException: If validation fails
        """
        return True

    def __call__(self, **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Execute tool with error handling.

        Returns:
            Tuple of (success, result, error_message)

        Note:
            Execution time tracking is handled by the ToolRegistry,
            not at the individual tool level.
        """
        try:
            # Validate parameters
            self.validate_parameters(**kwargs)

            # Execute tool
            result = self.execute(**kwargs)

            return True, result, None

        except ToolValidationException as e:
            return False, None, f"Validation error: {str(e)}"

        except ToolExecutionException as e:
            return False, None, f"Execution error: {str(e)}"

        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.get_metadata().name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.get_metadata().description


class SearchTool(BaseTool):
    """Base class for search-based tools."""

    @abstractmethod
    def search(self, query: str, **kwargs) -> str:
        """
        Search for information.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Search results as formatted string
        """
        pass

    def execute(self, **kwargs) -> Any:
        """Execute search."""
        query = kwargs.get("query", "")
        if not query:
            raise ToolValidationException("Query parameter is required")
        return self.search(query, **kwargs)


class AnalysisTool(BaseTool):
    """Base class for code analysis tools."""

    @abstractmethod
    def analyze(self, code: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        """
        Analyze code snippet.

        Args:
            code: Code to analyze
            language: Programming language
            **kwargs: Additional analysis parameters

        Returns:
            Analysis results as dictionary
        """
        pass

    def execute(self, **kwargs) -> Any:
        """Execute analysis."""
        code = kwargs.get("code", "")
        if not code:
            raise ToolValidationException("Code parameter is required")
        language = kwargs.get("language", "python")
        return self.analyze(code, language, **kwargs)


class DocumentationTool(BaseTool):
    """Base class for documentation retrieval tools."""

    @abstractmethod
    def get_documentation(
        self,
        method_name: str,
        language: str = "python",
        **kwargs
    ) -> str:
        """
        Retrieve documentation for a method/function.

        Args:
            method_name: Name of method/function
            language: Programming language
            **kwargs: Additional parameters

        Returns:
            Documentation as formatted string
        """
        pass

    def execute(self, **kwargs) -> Any:
        """Execute documentation retrieval."""
        method_name = kwargs.get("method_name", "")
        if not method_name:
            raise ToolValidationException("method_name parameter is required")
        language = kwargs.get("language", "python")
        return self.get_documentation(method_name, language, **kwargs)
