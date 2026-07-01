def signal_function(state: str, action: str, next_state: str) -> float:
    # Initialize Q-value estimate
    q_value = 0.0
    
    # Convert to lowercase for case-insensitive matching
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    state_lower = state.lower()
    
    # Check if goal appears to be achieved in next_state (highest priority)
    goal_indicators = [
        'success', 'completed', 'added', 'sent', 'created', 'updated', 
        'saved', 'done', 'finished', 'confirmed', 'submitted', 'posted',
        'event created', 'message sent', 'task added', 'appointment made',
        'goal achieved', 'task complete', 'form submitted'
    ]
    for indicator in goal_indicators:
        if indicator in next_state_lower:
            q_value = 0.95
            break
    
    # Check for error states that reduce value
    error_indicators = [
        'error', 'failed', 'invalid', 'denied', 'rejected', 
        'not found', 'unavailable', 'timeout', 'exception', 'alert'
    ]
    for indicator in error_indicators:
        if indicator in next_state_lower:
            q_value = max(q_value, 0.0)
            break
    
    # Evaluate action productivity
    if 'click' in action_lower or 'fill' in action_lower:
        # Productive actions get base value
        q_value = max(q_value, 0.25)
    elif 'press' in action_lower:
        # Key press can be productive
        q_value = max(q_value, 0.2)
    elif 'scroll' in action_lower:
        # Scrolling is exploratory
        q_value = max(q_value, 0.15)
    elif 'noop' in action_lower:
        # No-op is generally not productive
        q_value = max(q_value, 0.05)
    
    # Check for progress indicators in next_state
    progress_indicators = [
        'step', 'progress', 'remaining', 'current', 'next',
        'field', 'input', 'form', 'button', 'menu'
    ]
    for indicator in progress_indicators:
        if indicator in next_state_lower:
            q_value = max(q_value, 0.35)
            break
    
    # Check if state shows interaction elements (interactive state is good)
    interactive_indicators = ['bid', 'clickable', 'interactive', 'button', 'link']
    for indicator in interactive_indicators:
        if indicator in next_state_lower:
            q_value = max(q_value, 0.4)
            break
    
    # Penalize if next_state is very similar to state (no progress)
    if len(state_lower) > 0 and len(next_state_lower) > 0:
        # Simple similarity check - if states are very similar, action didn't help
        common_words = set(state_lower.split()) & set(next_state_lower.split())
        if len(common_words) > 0:
            similarity = len(common_words) / max(len(state_lower.split()), 1)
            if similarity > 0.9:
                q_value = max(q_value, 0.1)
    
    # Bonus for specific app-related success patterns
    app_success_patterns = [
        'calendar', 'todo', 'messenger', 'maps', 'code',
        'event', 'message', 'task', 'location', 'file'
    ]
    for pattern in app_success_patterns:
        if pattern in next_state_lower:
            q_value = max(q_value, 0.45)
            break
    
    # Ensure Q-value is in valid range [0, 1]
    return max(0.0, min(1.0, q_value))