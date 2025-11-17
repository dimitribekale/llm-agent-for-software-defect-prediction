"""Test 3: Security validation - block code execution attacks."""

def test_blocks_import_attack(doc_tool):
    """Test that __import__ expressions are blocked."""
    result = doc_tool.get_documentation(
        "__import__('os').system('ls')",
        language="python"
    )
    assert "Invalid" in result or "unsafe" in result
    assert "Documentation for" not in result

def test_blocks_dangerous_module(doc_tool):
    """Test that dangerous modules like 'os' are blocked."""
    result = doc_tool.get_documentation(
        "os.system",
        language="python"
    )
    assert "not in the safe module list" in result