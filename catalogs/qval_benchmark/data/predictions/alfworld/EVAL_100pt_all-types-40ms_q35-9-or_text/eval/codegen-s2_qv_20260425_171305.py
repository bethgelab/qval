def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    error_indicators = ['error', 'fail', 'cannot', 'unable', 'failed', 'blocked', 'impossible', 'refused']
    success_indicators = ['success', 'done', 'completed', 'finished', 'task complete', 'goal reached']
    progress_indicators = ['placed', 'cleaned', 'picked', 'put', 'moved', 'in the', 'on the', 'to the']
    
    next_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    for indicator in error_indicators:
        if indicator in next_lower:
            q_value -= 0.5
            break
    
    for indicator in success_indicators:
        if indicator in next_lower:
            q_value += 0.8
            break
    
    new_progress = 0
    for indicator in progress_indicators:
        if indicator in next_lower and indicator not in state_lower:
            new_progress += 1
    
    if new_progress > 0:
        q_value += 0.3 * new_progress
    
    if 'goal' in next_lower:
        q_value += 0.2
    
    if 'clean' in next_lower and 'dirty' not in next_lower:
        q_value += 0.15
    
    if 'pick' in action_lower or 'put' in action_lower or 'move' in action_lower:
        q_value += 0.05
    
    q_value = max(0.0, min(1.0, q_value))
    return q_value