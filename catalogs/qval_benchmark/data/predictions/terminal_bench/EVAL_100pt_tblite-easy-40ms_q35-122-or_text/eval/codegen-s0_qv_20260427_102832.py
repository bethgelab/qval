def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value estimate
    q_value = 0.0
    
    # Check for success/completion indicators in next_state
    success_patterns = ['success', 'completed', 'passed', 'test passed', 'verified', 
                        'done', 'finished', 'correct', 'ok', 'all tests passed']
    for pattern in success_patterns:
        if pattern in next_state.lower():
            q_value += 0.5
            break
    
    # Check for error/failure indicators in next_state
    error_patterns = ['error', 'failed', 'not found', 'permission denied', 
                      'command not found', 'syntax error', 'no such file',
                      'denied', 'incorrect', 'mismatch', 'fail']
    for pattern in error_patterns:
        if pattern in next_state.lower():
            q_value -= 0.2
            break
    
    # Check if action is a meaningful command (not empty, not comment)
    action_stripped = action.strip() if action else ""
    if action_stripped and len(action_stripped) > 0:
        if not action_stripped.startswith('#') and len(action_stripped) > 1:
            q_value += 0.05  # Small bonus for taking action
    
    # Check for progress indicators - output length change
    if next_state and state:
        next_len = len(next_state)
        state_len = len(state)
        if next_len > state_len + 20:
            q_value += 0.05  # More output suggests command executed
        elif next_len < state_len - 20:
            q_value -= 0.05  # Less output might indicate issue
    
    # Check for directory or file changes (progress markers)
    progress_markers = ['cd ', 'ls ', 'cat ', 'mkdir ', 'cp ', 'mv ', 'rm ', 
                        'find ', 'grep ', 'chmod ', 'python', 'bash', 'sh ']
    for marker in progress_markers:
        if marker in action.lower():
            q_value += 0.03
            break
    
    # Penalize empty or trivial actions
    if not action_stripped or len(action_stripped) < 2:
        q_value -= 0.1
    
    # Check if we're approaching the step limit (implicit urgency)
    # This is a heuristic - if state mentions step count or similar
    urgency_patterns = ['step', 'limit', 'remaining', 'timeout']
    for pattern in urgency_patterns:
        if pattern in next_state.lower():
            q_value -= 0.02  # Slight penalty for time pressure
    
    # Bound the Q-value to reasonable range [-1.0, 1.0]
    q_value = max(-1.0, min(1.0, q_value))
    
    return float(q_value)