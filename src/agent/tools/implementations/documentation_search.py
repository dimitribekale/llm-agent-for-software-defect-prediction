"""
Documentation search tool for retrieving official language documentation.
"""
import re
import inspect
import builtins
import importlib
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
        self.safe_modules = {
            'str', 'int', 'float', 'bool', 'list', 'dict',
            'set', 'tuple', 'bytes', 'math', 'json', 're',
            'datetime', 'collections', 'itertools',
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
        
    def _is_safe_identifier(self, name:str) -> bool:
        """
        Validate that the name is a safe Python identifier.
        
        Only allows:
        - Letters, numbers, underscores, and dots
        - No special characters or expressions
        - No dunder methods that could be dangerous
        
        Args:
            name: The method/module name to validate
            
        Returns:
            True if safe, False otherwise
        """
        if not name or not isinstance(name, str):
            return False
        
        # Prevent long suspicious inputs.
        if len(name)> 200:
            return
        
        # Only allow alphanumeric, underscore, and dot
        # This prevents expressions like __import__('os').system('...')
        if not re.match(r'^[a-zA-Z0-9_.]+$', name):
            return False
        
        dangerous_patterns = [
            '__import__',     # Dynamic imports
            'eval',
            'exec',
            'compile',
            'open',            # File operations
            '__',              # Double underscore methods
        ]
        for pattern in dangerous_patterns:
            if pattern in name.lower():
                return False
            
        return True

    def _get_safe_builtin_modules(self) -> set:
        """
        Get a set of safe Python built-in modules.
        
        Returns:
            Set of safe module names
        """
        return {
            'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'bytes', 'bytearray', 'frozenset', 'complex', 'math', 'random',
            'datetime', 'json', 're', 'collections', 'itertools', 'functools',
            'typing', 'enum', 'dataclasses', 'string'
        }

    def _search_python(self, method_name: str) -> str:
        """
        Search Python documentation using safe methods only.

        Args:
            method_name: Python method/class/module name

        Returns:
            Documentation text
        """
        if not self._is_safe_identifier(method_name):
            return (
              f"Invalid or potentially unsafe identifier:'{method_name}'\n\n"
              f"Only simple module and method names are allowed.\n"
              f"Forbidden: expressions, special characters, __dunder__ methods"
            )
        
        # Extract base module name
        parts = method_name.split('.')
        base_module = parts[0]

        # Check that it's a safe built-in type or module
        safe_modules = self._get_safe_builtin_modules()

        if base_module not in safe_modules:
            return (
                f"Module '{base_module}' is not in the safe module list.\n\n"
                f"For security reasons, only built-in Python modules are supported.\n"
                f"Supported modules: {', '.join(sorted(safe_modules))}\n\n"
                f"For other modules, please refer to: {self.official_docs['python']}\n"
                f"Search for: {method_name}"
            )
        
        try:
            obj = None

            if hasattr(builtins, base_module):
                obj = getattr(builtins, base_module)

                # If there are more parts
                for part in parts[1:]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        return f"'{method_name}' not found in built-in types."
            else:
                # Try importing the module (only if it's in safe list)
                try:
                    module = importlib.import_module(base_module)
                    obj = module

                    # Traverse the rest of the path
                    for part in parts[1:]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            return f"'{method_name}' not found."
                except ImportError:
                    return f"Module '{base_module}' could not be imported."
            
            # Get documentation using inspect
            if obj is not None:
                doc = inspect.getdoc(obj)
                if doc:
                    # Format 
                    result = f"Documentation for '{method_name}':\n\n{doc}\n\n"

                    # Add signature if available
                    try:
                        if callable(obj):
                            signature = inspect.signature(obj)
                            result += f"Signature: {method_name}{signature}\n"
                    except(ValueError, TypeError):
                        pass

                    return result
                
                else:
                    return f"No documentation found for '{method_name}'."
                
        except Exception as e:
            return (
                f"Could not retrieve documentation for '{method_name}'.\n\n"
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
