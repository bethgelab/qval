def signal_function(state: str) -> float:
    """
    Estimate state-value based on analysis of state representation.
    Returns a float between 0 and 1 representing estimated probability of success.
    """
    import re
    
    # Base value - start with a reasonable prior
    value = 0.3
    
    # Check for success indicators
    state_lower = state.lower()
    
    # Success/completion indicators
    success_patterns = ['done', 'success', 'completed', 'added', 'created', 'sent', 'saved', 'submitted']
    for pattern in success_patterns:
        if pattern in state_lower:
            value = min(value + 0.35, 1.0)
            break
    
    # Error/failure indicators (reduce value)
    error_patterns = ['error', 'failed', 'invalid', 'required', 'missing', 'warning']
    for pattern in error_patterns:
        if pattern in state_lower:
            value = max(value - 0.25, 0.0)
            break
    
    # Form completion indicators - filled fields suggest progress
    if 'value=' in state or re.search(r'filled', state_lower):
        value = min(value + 0.1, 1.0)
    
    # Check for interactive elements that suggest we're on the right page
    if 'button' in state_lower or 'submit' in state_lower or 'send' in state_lower or 'save' in state_lower:
        value = min(value + 0.1, 1.0)
    
    # Check for navigation elements (being on correct page is good)
    if 'nav' in state_lower or 'menu' in state_lower or 'tab' in state_lower:
        value = min(value + 0.05, 1.0)
    
    # Penalize states with many elements but no clear action (might be stuck)
    bid_count = state.count('bid')
    if bid_count > 40:
        value = max(value - 0.1, 0.0)
    
    # Ensure value is in valid range
    return max(0.0, min(1.0, value))