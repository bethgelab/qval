def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Start with a neutral baseline
    q_value = 0.5
    
    # Detect task completion indicators - strong positive signal
    completion_patterns = [
        r'(?:complete|success|verified|passed|done|finished|correct)',
        r'(?:✓|✔|✅)',
        r'(?:result|output|answer|solution)',
        r'(?:test.*pass|assert.*ok|check.*ok)',
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value = 0.95
            break
    
    # Detect error/failure indicators - strong negative signal
    error_patterns = [
        r'(?:error|fail|failed|exception|traceback)',
        r'(?:not found|permission|denied|refused|invalid)',
        r'(?:wrong|incorrect|bad|malformed)',
        r'(?:test.*fail|assert.*fail|check.*fail)',
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value = min(q_value, 0.15)
            break
    
    # Reward progress indicators - moderate positive signal
    progress_patterns = [
        r'(?:step|progress|completed|done)',
        r'(?:\d+/\d+)',  # Progress like "3/5"
        r'(?:percent|%)',
        r'(?:\d+\.\d+)',  # Numeric results
        r'(?:created|wrote|generated|built)',
    ]
    
    for pattern in progress_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value = min(q_value, 0.85)
            break
    
    # Reward state change (action had observable effect)
    if next_state != state and len(next_state.strip()) > 0:
        q_value += 0.05
    
    # Penalize very short next_state (might indicate command failure)
    if len(next_state.strip()) < 5 and len(state.strip()) > 20:
        q_value -= 0.15
    
    # Reward meaningful action (non-trivial commands)
    action_lower = action.lower().strip()
    if len(action_lower) > 3 and not action_lower in ['ls', 'cd', 'pwd']:
        q_value += 0.02
    
    # Clamp to valid range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value