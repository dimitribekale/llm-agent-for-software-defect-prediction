"""
Documentation search tool for retrieving official language documentation.
"""

import pydoc
from typing import Any
from ..base import DocumentationTool, ToolMetadata, ToolValidationException


class DocumentationSearchTool(DocumentationTool):
    """
    Tool for searching official programming language documentation.

    Supports Python (via pydoc) and provides links for other languages.
    """

    def __init__(self):
        super().__init__()
        self.official_docs = {
            "python": "https://docs.python.org/3/library/",
            "java": "https://docs.oracle.com/javase/8/docs/api/",
            "c++": "https://en.cppreference.com/w/",
            "cpp": "https://en.cppreference.com/w/",
            "rust": "https://doc.rust-lang.org/std/",
            "c": "https://en.cppreference.com/w/c"
        }

    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name="documentation_search",
            description="Search official documentation for methods, functions, and language constructs",
            parameters={
                "method_name": "Name of method/function to look up (required)",
                "language": "Programming language (python, java, c++, rust, c)"
            },
            returns="Documentation text or links to official documentation",
            examples=[
                "method_name='list.append', language='python'",
                "method_name='malloc', language='c'"
            ]
        )

    def validate_parameters(self, **kwargs) -> bool:
        """Validate documentation search parameters."""
        method_name = kwargs.get("method_name", "")
        if not method_name or not isinstance(method_name, str):
            raise ToolValidationException("method_name must be a non-empty string")

        language = kwargs.get("language", "python")
        if not isinstance(language, str):
            raise ToolValidationException("language must be a string")

        return True

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
            Documentation text
        """
        method_name = method_name.strip()
        language = language.lower().strip()

        if language == "python":
            return self._search_python(method_name)
        else:
            return self._search_other(method_name, language)

    def _search_python(self, method_name: str) -> str:
        """
        Search Python documentation using pydoc.

        Args:
            method_name: Python method/class/module name

        Returns:
            Documentation text
        """
        try:
            # Try to get documentation
            doc = pydoc.render_doc(method_name, "Help on %s")
            return doc
        except Exception as e:
            # Fallback to help link
            return (
                f"Could not retrieve documentation for '{method_name}' using pydoc.\n"
                f"Error: {str(e)}\n\n"
                f"Please refer to: {self.official_docs['python']}\n"
                f"Search for: {method_name}"
            )

    def _search_other(self, method_name: str, language: str) -> str:
        """
        Provide documentation links for other languages.

        Args:
            method_name: Method/function name
            language: Programming language

        Returns:
            Documentation link and guidance
        """
        doc_link = self.official_docs.get(language)

        if doc_link:
            return (
                f"Documentation for '{method_name}' in {language.upper()}:\n\n"
                f"Official documentation: {doc_link}\n"
                f"Search for: {method_name}\n\n"
                f"Common issues to check:\n"
                f"- Correct parameter types and counts\n"
                f"- Return value handling\n"
                f"- Memory management (for C/C++)\n"
                f"- Ownership and borrowing (for Rust)\n"
                f"- Null/undefined handling\n"
                f"- Exception handling"
            )
        else:
            return (
                f"No documentation source configured for language: {language}\n"
                f"Supported languages: {', '.join(self.official_docs.keys())}"
            )
