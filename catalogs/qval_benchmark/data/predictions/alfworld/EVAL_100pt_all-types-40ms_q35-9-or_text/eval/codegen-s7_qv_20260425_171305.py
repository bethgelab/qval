import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    goal_complete_patterns = ['goal', 'done', 'complete', 'success', 'task.*complete']
    error_patterns = ['error', 'fail', 'cannot', 'unable', 'invalid', 'wrong']
    progress_patterns = ['moved', 'placed', 'picked', 'cleaned', 'opened', 'closed', 
                        'turned', 'put.*in', 'put.*on', 'placed.*in', 'placed.*on',
                        'successfully', 'completed']
    
    has_goal_next = any(re.search(p, next_state_lower) for p in goal_complete_patterns)
    has_error_next = any(re.search(p, next_state_lower) for p in error_patterns)
    has_progress_next = any(re.search(p, next_state_lower) for p in progress_patterns)
    
    has_goal_state = any(re.search(p, state_lower) for p in goal_complete_patterns)
    has_error_state = any(re.search(p, state_lower) for p in error_patterns)
    has_progress_state = any(re.search(p, state_lower) for p in progress_patterns)
    
    if has_goal_next and not has_error_next:
        score = 0.95
    elif has_error_next:
        score = -0.5
    elif has_progress_next and not has_progress_state:
        score = 0.35
    elif has_progress_next and has_progress_state:
        score = 0.15
    elif not has_progress_next and not has_progress_state:
        score = 0.0
    else:
        score = -0.1
    
    if action_lower in state_lower:
        score += 0.05
    
    return max(-1.0, min(1.0, score))