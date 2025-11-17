"""Test 1: Tool initialization."""

def test_tool_creates_successfully(doc_tool):
    """Verify the tool initializes with required attributes."""
    assert doc_tool is not None
    assert hasattr(doc_tool, 'safe_modules')
    assert 'str' in doc_tool.safe_modules
    assert 'list' in doc_tool.safe_modules