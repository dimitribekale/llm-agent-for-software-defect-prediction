import re

def _is_safe_identifier(name:str) -> bool:
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

def _get_safe_builtin_modules() -> set:
    """
    Get a set of safe Python built-in modules.
    
    Returns:
        Set of safe module names
    """
    return {
        'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
        'bytes', 'bytearray', 'frozenset', 'complex', 'match', 'random',
        'datetime', 'json', 're', 'collections', 'itertools', 'functools',
        'typing', 'enum', 'dataclasses', 'string'
    }