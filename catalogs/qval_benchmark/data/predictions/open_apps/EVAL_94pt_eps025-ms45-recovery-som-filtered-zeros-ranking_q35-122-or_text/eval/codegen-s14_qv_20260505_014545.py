def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for state-action-next_state transition in OpenApps environment.
    
    Heuristic estimation based on:
    1. Task completion signals in next_state
    2. Action effectiveness and appropriateness
    3. State change magnitude indicating progress
    4. Error/invalid state detection
    """
    import re
    
    q_value = 0.0
    
    # 1. Check for task completion in next_state - immediate high reward
    completion_patterns = [
        r'task.*completed', r'success', r'saved', r'added', 
        r'created', r'sent', r'published', r'done', r'confirmed',
        r'submitted', r'scheduled', r'message.*sent', r'event.*created',
        r'todo.*added', r'calendar.*added', r'message.*delivered'
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 1.0
    
    # 2. Analyze action effectiveness
    action_lower = action.lower()
    
    # Productive actions (click, fill, press) add value
    if re.search(r"click\(['\"]\d+['\"]", action_lower):
        q_value += 0.12
    elif re.search(r"fill\(['\"]\d+['\"]", action_lower):
        q_value += 0.15
    elif re.search(r"press\(['\"]\d+['\"]", action_lower):
        q_value += 0.08
    
    # Noop is low value
    if 'noop' in action_lower:
        q_value -= 0.05
    
    # Scroll can be productive for navigation
    if 'scroll' in action_lower:
        q_value += 0.03
    
    # 3. State change analysis - significant changes indicate progress
    state_len = len(state)
    next_state_len = len(next_state)
    state_diff = abs(next_state_len - state_len)
    
    if state_diff > 200:
        q_value += 0.25
    elif state_diff > 100:
        q_value += 0.15
    elif state_diff > 50:
        q_value += 0.08
    elif state_diff > 20:
        q_value += 0.04
    
    # 4. Look for progress indicators in next_state
    progress_indicators = [
        'loading', 'processing', 'saving', 'updating', 
        'changed', 'modified', 'updated', 'step', 'progress'
    ]
    
    for indicator in progress_indicators:
        if indicator in next_state.lower():
            q_value += 0.03
            break
    
    # 5. Check for error/invalid states - penalize
    error_patterns = [
        r'error', r'failed', r'invalid', r'not found',
        r'cannot', r'unable', r'missing', r'required',
        r'incorrect', r'wrong', r'invalid input'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value -= 0.20
            break
    
    # 6. Check for bid tags in next_state (interactive elements available)
    bid_count = len(re.findall(r"bid=['\"](\d+)['\"]", next_state))
    if bid_count > 0:
        q_value += 0.05  # More interactive elements = more options
    
    # 7. Check if next_state has form fields (fillable content)
    if re.search(r'<input|<textarea|type=["\']text', next_state, re.IGNORECASE):
        q_value += 0.05
    
    # Normalize Q-value to [0, 1] range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value