"""Test 2: Parameter validation."""
import pytest
from src.agent.tools.base import ToolValidationException

def test_valid_parameters(doc_tool):
    """Test that valid parameters pass validation."""
    result = doc_tool.validate_parameters(
        method_name="list.append",
        language="python"
    )
    assert result is True

def test_empty_method_name(doc_tool):
    """Test that empty method names raise exception."""
    with pytest.raises(ToolValidationException):
        doc_tool.validate_parameters(
            method_name="",
            language="python"
        )