"""
Shared test fixtures for the entire test suite.
"""
import pytest
from src.agent.tools.implementations.documentation_search import DocumentationSearchTool

@pytest.fixture
def doc_tool():
    """
    Create a DocumentationSearchTool instance for testing.

    This fixture provides a fresh instance for each test,
    ensuring test isolation.

    Returns:
        DocumentationSearchTool: Fresh tool instance
    """
    return DocumentationSearchTool()

@pytest.fixture
def safe_method_names():
    """
    Provide a list of SAFE method names for testing.

    Returns:
        list: Safe Python method/module names
    """
    return [
            "list.append",
            "str.split",
            "dict.get",
            "math.sqrt",
            "json.dumps",
            "re.match",
            "datetime.now",
    ]

@pytest.fixture
def malicious_method_names():
    """
    Provide a list of MALICIOUS method names that should be blocked.

    Returns:
        list: Dangerous inputs that could lead to code execution
    """
    return [
        "__import__('os').system('rm -rf /')",
        "eval('print(1)')",
        "exec('import os')",
        "compile('x=1', '', 'exec')",
        "__builtins__.__import__('os')",
        "os.system",  # os not in whitelist
        "subprocess.run",  # subprocess not in whitelist
        "open('/etc/passwd')",
    ]
