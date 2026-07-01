def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check for goal completion indicators in next_state
    goal_keywords = ['done', 'completed', 'success', 'saved', 'created', 'added', 'sent', 'updated', 'finished', 'event created', 'message sent', 'task added']
    next_state_lower = next_state.lower()
    
    # Check if goal appears to be achieved
    goal_achieved = any(keyword in next_state_lower for keyword in goal_keywords)
    
    if goal_achieved:
        return 1.0
    
    # Parse action type and assign base score
    action_lower = action.lower()
    action_score = 0.0
    
    if 'fill(' in action_lower:
        # Filling forms is highly productive in web tasks
        action_score = 0.45
    elif 'click(' in action_lower:
        # Clicking is fundamental for navigation
        action_score = 0.35
    elif 'press(' in action_lower:
        # Key presses are moderately productive
        action_score = 0.25
    elif 'scroll(' in action_lower:
        # Scrolling is less directly productive
        action_score = 0.15
    elif 'noop(' in action_lower:
        # No-op actions provide no progress
        action_score = 0.0
    
    # Check for meaningful state changes (progress indicator)
    state_change_score = 0.0
    if state != next_state:
        # State changed - likely progress
        state_change_score = 0.25
        # Check magnitude of change
        if len(state) > 0 and len(next_state) > 0:
            # Simple heuristic: larger changes may indicate more progress
            change_ratio = abs(len(next_state) - len(state)) / max(len(state), 1)
            if change_ratio > 0.1:
                state_change_score = 0.35
    
    # Look for progress indicators in next_state
    progress_keywords = ['new', 'updated', 'changed', 'modified', 'event', 'message', 'task', 'calendar', 'todo', 'item', 'entry', 'form', 'input', 'field', 'button', 'submit', 'save']
    has_progress = any(keyword in next_state_lower for keyword in progress_keywords)
    progress_score = 0.2 if has_progress else 0.0
    
    # Check for negative indicators (errors, failures)
    negative_keywords = ['error', 'failed', 'invalid', 'missing', 'required', 'cannot', 'unable', 'denied']
    has_negative = any(keyword in next_state_lower for keyword in negative_keywords)
    negative_penalty = 0.3 if has_negative else 0.0
    
    # Check if action matches likely productive patterns for web apps
    bid_pattern = re.search(r'bid\s*=\s*[\'"](\d+)[\'"]', action_lower) or re.search(r'click\([\'"](\d+)[\'"]', action)
    has_bid = bid_pattern is not None
    bid_score = 0.1 if has_bid else 0.0
    
    # Combine scores with weights
    q_value = action_score + state_change_score + progress_score + bid_score - negative_penalty
    
    # Ensure Q-value is in valid range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)