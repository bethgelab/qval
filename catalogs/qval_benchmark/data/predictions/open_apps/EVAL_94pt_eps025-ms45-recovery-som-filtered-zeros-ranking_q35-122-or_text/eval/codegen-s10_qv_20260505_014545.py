def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Start with base Q-value for non-terminal state
    q_value = 0.1
    
    # Check for task completion indicators in next_state
    completion_patterns = [
        r'success', r'saved', r'added', r'created', r'completed',
        r'sent', r'published', r'confirmed', r'done', r'task\s+completed',
        r'event\s+added', r'message\s+sent', r'todo\s+added',
        r'calendar', r'message\s+delivered', r'event\s+created'
    ]
    
    completion_score = 0
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            completion_score += 1
    
    # Boost Q-value based on completion signals
    if completion_score >= 3:
        q_value = 0.95
    elif completion_score >= 2:
        q_value = 0.8
    elif completion_score >= 1:
        q_value = 0.6
    
    # Check for error or failure indicators
    error_patterns = [
        r'error', r'failed', r'invalid', r'not\s+found', 
        r'cannot', r'unable', r'required'
    ]
    
    error_score = 0
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            error_score += 1
    
    # Reduce Q-value for error indicators
    if error_score >= 2:
        q_value *= 0.3
    elif error_score >= 1:
        q_value *= 0.6
    
    # Evaluate action quality
    action_lower = action.lower()
    
    # Productive actions get bonus
    if re.search(r"click\(['\"]\w+['\"]\)", action_lower):
        q_value *= 1.15
    elif re.search(r"fill\(['\"]\w+['\"]", action_lower):
        q_value *= 1.1
    elif re.search(r"press\(['\"]\w+['\"]", action_lower):
        q_value *= 1.05
    
    # Non-productive actions get penalty
    if 'noop' in action_lower:
        q_value *= 0.5
    elif 'scroll' in action_lower:
        q_value *= 0.85
    
    # Check if state changed meaningfully (progress indicator)
    # More unique elements in next_state suggests progress
    next_elements = re.findall(r"bid['\"]?[:=]\s*['\"]?(\d+)", next_state)
    state_elements = re.findall(r"bid['\"]?[:=]\s*['\"]?(\d+)", state)
    
    if len(set(next_elements)) > len(set(state_elements)):
        q_value *= 1.1
    
    # Check for form completion progress
    form_indicators = [
        r'input\s*=\s*["\']', r'value=["\']', r'filled', r'typed'
    ]
    for pattern in form_indicators:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value *= 1.05
            break
    
    # Ensure Q-value stays in valid [0, 1] range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value