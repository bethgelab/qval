def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.5
    
    success_indicators = ['exit code 0', 'success', 'ok', 'completed', 'done', 'no error', 'passed', 'true', 'yes']
    for pattern in success_indicators:
        if pattern.lower() in next_state.lower():
            q_value += 0.2
            break
    
    error_indicators = ['error', 'failed', 'exception', 'traceback', 'connection refused', 'permission denied', 'killed', 'timeout', 'false', 'no']
    for pattern in error_indicators:
        if pattern.lower() in next_state.lower():
            q_value -= 0.25
            break
    
    progress_indicators = ['created', 'wrote', 'saved', 'exported', 'generated', 'compiled', 'built', 'installed', 'copied', 'moved', 'deleted', 'removed', 'updated', 'modified']
    for pattern in progress_indicators:
        if pattern.lower() in next_state.lower():
            q_value += 0.1
            break
    
    if action and action.strip():
        q_value += 0.05
    
    if state and 'error' in state.lower() and 'error' not in next_state.lower():
        q_value += 0.1
    
    q_value = max(0.0, min(1.0, q_value))
    return q_value