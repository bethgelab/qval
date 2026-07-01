import re

def signal_function(state: str, action: str, next_state: str) -> float:
    base_value = 0.5
    
    error_indicators = ['error', 'failed', 'exception', 'traceback', 'fatal', 'invalid', 'denied', 'refused']
    success_indicators = ['success', 'completed', 'done', 'created', 'wrote', 'generated', 'exported', 'saved', 'wrote', 'finished']
    progress_indicators = ['added', 'modified', 'updated', 'changed', 'copied', 'moved', 'installed', 'built', 'compiled']
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    state_errors = sum(1 for pattern in error_indicators if pattern in state_lower)
    state_successes = sum(1 for pattern in success_indicators if pattern in state_lower)
    state_progress = sum(1 for pattern in progress_indicators if pattern in state_lower)
    
    next_errors = sum(1 for pattern in error_indicators if pattern in next_state_lower)
    next_successes = sum(1 for pattern in success_indicators if pattern in next_state_lower)
    next_progress = sum(1 for pattern in progress_indicators if pattern in next_state_lower)
    
    error_change = next_errors - state_errors
    success_change = next_successes - state_successes
    progress_change = next_progress - state_progress
    
    net_improvement = success_change + progress_change - error_change
    
    if net_improvement > 0:
        q_value = base_value + 0.15 * net_improvement
    elif net_improvement < 0:
        q_value = base_value - 0.15 * abs(net_improvement)
    else:
        q_value = base_value
    
    if action and action.strip():
        action_lower = action.lower()
        if 'rm' in action_lower and 'force' not in action_lower:
            q_value -= 0.1
        if 'chmod' in action_lower or 'chown' in action_lower:
            q_value += 0.05
        if 'cat' in action_lower:
            q_value += 0.02
    
    if next_state and 'prompt' in next_state_lower and 'error' not in next_state_lower:
        q_value += 0.05
    
    q_value = max(0.0, min(1.0, q_value))
    return q_value