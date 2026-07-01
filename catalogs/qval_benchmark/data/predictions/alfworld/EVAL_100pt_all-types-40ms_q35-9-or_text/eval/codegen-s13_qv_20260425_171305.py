def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    q_value = 0.5
    
    progress_indicators = ['completed', 'done', 'success', 'goal achieved', 'task done', 'finished']
    failure_indicators = ['failed', 'error', 'invalid', 'cannot', 'blocked', 'unable']
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    state_progress = sum(1 for ind in progress_indicators if ind in state_lower)
    next_state_progress = sum(1 for ind in progress_indicators if ind in next_state_lower)
    
    state_failure = sum(1 for ind in failure_indicators if ind in state_lower)
    next_state_failure = sum(1 for ind in failure_indicators if ind in next_state_lower)
    
    if next_state_progress > state_progress:
        q_value += 0.25
    elif next_state_progress < state_progress:
        q_value -= 0.15
    
    if next_state_failure > state_failure:
        q_value -= 0.35
    elif next_state_failure < state_failure:
        q_value += 0.1
    
    navigation_keywords = ['go to', 'move to', 'walk to', 'navigate to', 'head to', 'reach']
    if any(kw in action.lower() for kw in navigation_keywords):
        q_value += 0.1
    
    manipulation_keywords = ['take', 'put', 'clean', 'wash', 'dry', 'move', 'carry']
    if any(kw in action.lower() for kw in manipulation_keywords):
        q_value += 0.05
    
    if 'object' in state_lower and 'object' in next_state_lower:
        q_value += 0.1
    
    if 'in hand' in state_lower and 'in hand' not in next_state_lower:
        q_value += 0.15
    
    if 'in ' in next_state_lower and 'in ' not in state_lower:
        q_value += 0.05
    
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value