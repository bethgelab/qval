def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check for immediate success indicators in next_state
    success_patterns = [
        'successfully', 'task completed', 'congratulations', 
        'reward', 'done', 'complete', 'finished', 'goal'
    ]
    for pattern in success_patterns:
        if pattern.lower() in next_state.lower():
            return 1.0
    
    # Check for failure or unproductive states
    failure_patterns = [
        'cannot', 'invalid', 'nothing', 'no', 'wrong',
        'failed', 'error', 'not found', 'not in front of'
    ]
    for pattern in failure_patterns:
        if pattern.lower() in next_state.lower():
            return 0.0
    
    # Estimate base value based on action type
    action_lower = action.lower()
    
    # High-value actions that directly manipulate objects
    manipulation_actions = ['take', 'pick', 'put', 'place', 'clean', 'heat', 'cook', 'open', 'close', 'fill', 'empty']
    nav_actions = ['go to', 'walk to', 'move to', 'navigate', 'look']
    
    if any(act in action_lower for act in manipulation_actions):
        base_progress = 0.4
    elif any(nav in action_lower for nav in nav_actions):
        base_progress = 0.2
    else:
        base_progress = 0.1
    
    # Check next_state for progress indicators
    progress_boost = 0.0
    
    # Object found or picked up
    if any(ind in next_state.lower() for ind in ['holding', 'carrying', 'found', 'picked up', 'take']):
        progress_boost += 0.25
    
    # Object placed at target
    if any(ind in next_state.lower() for ind in ['put on', 'placed', 'put in', 'put on top', 'put next to']):
        progress_boost += 0.35
    
    # Object state changed (cleaned, heated, etc.)
    if any(ind in next_state.lower() for ind in ['cleaned', 'heated', 'cooked', 'filled', 'emptied']):
        progress_boost += 0.2
    
    # Check for setbacks or wasted actions
    setback_penalty = 0.0
    if any(ind in next_state.lower() for ind in ['back', 'return', 'again', 'repeat', 'already']):
        setback_penalty += 0.15
    
    # Check if we're in a different room (navigation progress)
    if any(ind in next_state.lower() for ind in ['room', 'kitchen', 'bedroom', 'bathroom', 'living room', 'dining room']):
        progress_boost += 0.1
    
    # Calculate final Q-value
    q_value = base_progress + progress_boost - setback_penalty
    
    # Clamp to [0, 1] range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value