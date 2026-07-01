import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on state analysis without simulation.
    
    Higher values indicate better expected returns.
    """
    score = 0.0
    
    # Check if next_state indicates goal completion
    completion_indicators = ['goal', 'done', 'success', 'complete', 'task', 'finished', 'succeed']
    if any(kw in next_state.lower() for kw in completion_indicators):
        score += 0.9
    
    # Check if state already indicates goal completion (already at goal)
    if any(kw in state.lower() for kw in completion_indicators):
        score += 0.5
    
    # Check for error messages (bad action)
    error_indicators = ['error', 'cannot', 'invalid', 'not', 'already', 'failed', 'wrong']
    if any(kw in next_state.lower() for kw in error_indicators):
        score -= 0.6
    
    # Check if state changed (action had effect)
    if state != next_state:
        score += 0.1
    
    # Check for progress keywords in next_state that weren't in state
    progress_keywords = ['holding', 'on', 'in', 'placed', 'cleaned', 'opened', 'closed', 'put']
    state_progress = sum(1 for kw in progress_keywords if kw in state.lower())
    next_progress = sum(1 for kw in progress_keywords if kw in next_state.lower())
    if next_progress > state_progress:
        score += 0.2
    
    # Check if action is mentioned in next_state (successful execution)
    if action.lower() in next_state.lower():
        score += 0.1
    
    # Check if location changed (movement toward goal)
    if 'in the' in next_state.lower() and 'in the' not in state.lower():
        score += 0.15
    
    # Cap the score between 0 and 1
    score = max(0.0, min(1.0, score))
    
    return score