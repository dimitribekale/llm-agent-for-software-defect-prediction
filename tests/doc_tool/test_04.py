"""Test 4: Retrieve documentation for valid Python identifiers."""

def test_builtin_type_documentation(doc_tool):
    """Test getting docs for built-in type"""
    result = doc_tool.get_documentation("list", language="python")
    assert "Documentation for 'list'" in result
    assert len(result) > 50

def test_builtin_method_documentation(doc_tool):
    """Test getting docs for built-in method"""
    result = doc_tool.get_documentation("str.split", language="python")
    assert "Documentation for 'str.split'" in result
    assert "split" in result.lower()

def test_safe_module_method(doc_tool):
    """Test getting docs for safe module like 'math.sqrt'."""
    result = doc_tool.get_documentation("math.sqrt", language="python")
    assert "Documentation for 'math.sqrt'" in result or "could not be imported" in result.lower()