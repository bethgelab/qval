def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    # Initialize Q-value estimate
    q_value = 0.0
    
    # Progress indicators - actions that typically advance the task
    progress_keywords = [
        'picked', 'put', 'clean', 'cleaned', 'move', 'moved',
        'go to', 'go', 'open', 'close', 'take', 'drop',
        'hold', 'cleaning', 'cleaned', 'placed', 'putting'
    ]
    
    # Negative indicators - actions that might hinder progress
    negative_keywords = [
        'cannot', 'unable', 'failed', 'error', 'wrong',
        'already', 'full', 'occupied', 'blocked', 'impossible'
    ]
    
    # Goal completion indicators
    completion_keywords = [
        'done', 'complete', 'finished', 'success', 'goal',
        'task completed', 'all', 'everything', 'ready'
    ]
    
    # State change analysis
    state_changed = state != next_state
    
    # Check if action appears to be valid and productive
    action_lower = action.lower()
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Count progress indicators in next state
    next_progress_count = sum(1 for kw in progress_keywords if kw in next_state_lower)
    current_progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    
    # Check for negative indicators
    has_negative = any(kw in state_lower or kw in next_state_lower for kw in negative_keywords)
    has_completion = any(kw in state_lower or kw in next_state_lower for kw in completion_keywords)
    
    # Base Q-value on whether state changed (action had effect)
    if state_changed:
        base_value = 0.3
    else:
        base_value = 0.0
    
    # Adjust based on progress indicators in next state
    if next_progress_count > 0:
        base_value += min(0.3, next_progress_count * 0.1)
    
    # Penalize if negative indicators present
    if has_negative:
        base_value -= 0.3
    
    # Bonus for completion indicators
    if has_completion:
        base_value += 0.4
    
    # Consider action type
    if 'go' in action_lower and 'to' in action_lower:
        base_value += 0.1  # Navigation is usually safe
    elif 'pick' in action_lower or 'drop' in action_lower or 'clean' in action_lower:
        base_value += 0.1  # Object manipulation is usually productive
    
    # Normalize to reasonable range
    q_value = max(0.0, min(1.0, base_value))
    
    # If state didn't change, reduce Q-value
    if not state_changed and q_value > 0:
        q_value *= 0.5
    
    return q_value