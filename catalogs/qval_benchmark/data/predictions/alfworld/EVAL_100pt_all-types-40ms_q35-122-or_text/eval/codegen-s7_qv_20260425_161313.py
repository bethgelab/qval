def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check if task is complete in next_state
    completion_patterns = [
        r'task completed',
        r'you have completed',
        r'the task is complete',
        r'congratulations',
        r'task success',
        r'you won',
        r'episode done',
        r'task done',
        r'you have finished',
        r'task finished'
    ]
    
    is_complete = False
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            is_complete = True
            break
    
    if is_complete:
        return 1.0
    
    # Check for error/invalid action indicators
    error_patterns = [
        r'nothing to',
        r'cannot',
        r'not possible',
        r'invalid',
        r'error',
        r'failed',
        r'does not exist',
        r'you cannot',
        r'there is no',
        r'no such',
        r'not found',
        r'cannot find',
        r'nothing in'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.0
    
    # Analyze action type
    action_lower = action.lower()
    
    # High-value actions (directly contribute to goal)
    high_value_actions = ['pickup', 'put', 'place', 'clean', 'heat', 'cool', 'toggle', 'open', 'close']
    
    # Medium-value actions (navigation, exploration)
    medium_value_actions = ['go', 'walk', 'move', 'look', 'examine']
    
    # Low-value actions (redundant or uncertain)
    low_value_actions = ['think', 'wait', 'idle', 'help']
    
    if any(action in action_lower for action in high_value_actions):
        base_q = 0.4
    elif any(action in action_lower for action in medium_value_actions):
        base_q = 0.2
    elif any(action in action_lower for action in low_value_actions):
        base_q = 0.1
    else:
        base_q = 0.15
    
    # Check if state changed meaningfully
    state_changed = len(next_state.strip()) > 0 and len(state.strip()) > 0
    
    # Bonus if state changed significantly (action had effect)
    if state_changed:
        base_q *= 1.2
    
    # Check for progress indicators in next_state
    progress_indicators = [
        r'you are now at',
        r'you have picked up',
        r'you have put',
        r'you have placed',
        r'you have cleaned',
        r'you have heated',
        r'you have cooled',
        r'you have opened',
        r'you have closed',
        r'you have toggled',
        r'you are holding',
        r'you are carrying',
        r'you see',
        r'you find',
        r'you are in'
    ]
    
    for pattern in progress_indicators:
        if re.search(pattern, next_state, re.IGNORECASE):
            base_q *= 1.3
            break
    
    # Cap at reasonable value (not complete yet)
    return min(0.8, base_q)