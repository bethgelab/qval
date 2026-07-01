def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Initialize base Q-value
    q_value = 0.1
    
    # Check for task completion indicators in state
    completion_patterns = [
        r'task.*complete',
        r'task.*success',
        r'task.*done',
        r'successfully.*completed',
        r'event.*created',
        r'message.*sent',
        r'todo.*added',
        r'item.*saved',
        r'confirmation',
        r'confirmed',
    ]
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # If completion detected, high Q-value
    for pattern in completion_patterns:
        if re.search(pattern, state_lower) or re.search(pattern, next_state_lower):
            q_value = 0.95
            break
    
    # Analyze action type for productivity
    action_lower = action.lower()
    
    if 'fill' in action_lower and re.search(r"bid['\"]?\s*[0-9]+", action_lower):
        # Filling forms is productive progress
        q_value = max(q_value, 0.6)
    elif 'click' in action_lower and re.search(r"bid['\"]?\s*[0-9]+", action_lower):
        # Clicking on specific elements is productive
        q_value = max(q_value, 0.5)
    elif 'press' in action_lower and 'enter' in action_lower:
        # Submitting with enter is productive
        q_value = max(q_value, 0.55)
    elif 'press' in action_lower:
        # Other key presses
        q_value = max(q_value, 0.3)
    elif 'scroll' in action_lower:
        # Scrolling can be productive for navigation
        q_value = max(q_value, 0.25)
    elif 'noop' in action_lower:
        # No-op is generally unproductive
        q_value = max(q_value, 0.05)
    
    # Check for bid targeting (indicates specific element interaction)
    bid_pattern = r"bid['\"]?\s*[0-9]+"
    if re.search(bid_pattern, action_lower):
        bid_count = len(re.findall(bid_pattern, action_lower))
        if bid_count >= 1:
            q_value = max(q_value, 0.35)
    
    # Analyze state for available interactive elements
    element_count = len(re.findall(r"bid['\"]?\s*[0-9]+", state))
    if element_count >= 15:
        # Many elements means good navigation options
        q_value = max(q_value, 0.35)
    elif element_count >= 8:
        q_value = max(q_value, 0.3)
    elif element_count >= 3:
        q_value = max(q_value, 0.25)
    
    # Check for progress indicators in state
    progress_patterns = [
        r'page.*loaded',
        r'form.*ready',
        r'input.*visible',
        r'button.*available',
        r'element.*found',
        r'navigation.*complete',
        r'view.*updated',
    ]
    
    for pattern in progress_patterns:
        if re.search(pattern, state_lower):
            q_value = max(q_value, 0.4)
            break
    
    # Check if next_state shows meaningful change
    if len(next_state) > len(state) * 1.05:
        # State grew significantly, likely progress
        q_value = max(q_value, 0.45)
    
    # Check for error or invalid indicators
    error_patterns = [
        r'error',
        r'invalid',
        r'failed',
        r'not.*found',
        r'cannot',
        r'unable',
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, state_lower) or re.search(pattern, next_state_lower):
            q_value = min(q_value, 0.15)
            break
    
    # Penalize empty or minimal states
    if len(state) < 200:
        q_value = max(q_value, 0.05)
    
    # Ensure Q-value is in valid range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value