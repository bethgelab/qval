def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Extract action type
    action_lower = action.lower()
    if '(' in action_lower:
        action_type = action_lower.split('(')[0]
    else:
        action_type = action_lower
    
    # Action effectiveness by type
    action_effectiveness = {
        'fill': 0.25,
        'click': 0.25,
        'press': 0.20,
        'scroll': 0.10,
        'noop': -0.05
    }
    action_score = action_effectiveness.get(action_type, 0.0)
    
    # Goal completion indicators in next state
    goal_keywords = ['success', 'completed', 'done', 'saved', 'added', 'created', 
                     'message sent', 'event added', 'task completed', 'goal reached',
                     'verified', 'confirmed', 'finished', 'achieved', 'target reached']
    next_lower = next_state.lower()
    goal_matches = sum(1 for kw in goal_keywords if kw in next_lower)
    
    # State change detection
    state_changed = state != next_state
    
    # Bid tag analysis (interactive elements)
    bids_state = re.findall(r'\d+', state)
    bids_next = re.findall(r'\d+', next_state)
    bid_diff = len(bids_next) - len(bids_state)
    
    # Progress estimation
    progress_score = 0.0
    if goal_matches > 0:
        progress_score = 0.5 + min(0.4, goal_matches * 0.1)
    elif state_changed:
        progress_score = 0.15
    elif bid_diff > 0:
        progress_score = 0.05
    
    # Combined Q-value
    q_value = action_score + progress_score
    
    # Clamp range
    q_value = max(-0.3, min(1.0, q_value))
    
    return q_value