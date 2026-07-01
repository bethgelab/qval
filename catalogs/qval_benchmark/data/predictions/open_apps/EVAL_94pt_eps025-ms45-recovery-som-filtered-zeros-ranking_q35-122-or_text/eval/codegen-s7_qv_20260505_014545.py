def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    q_value = 0.0
    
    # Analyze state changes to detect meaningful progress
    state_len = len(state)
    next_len = len(next_state)
    state_change = abs(next_len - state_len)
    
    # Significant state change indicates meaningful action
    if state_change > 100:
        q_value += 0.25
    elif state_change > 50:
        q_value += 0.15
    elif state_change > 20:
        q_value += 0.05
    
    # Count bid tags (interactive elements) to assess complexity
    bid_pattern = r"bid['\"]?\s*[:=]\s*['\"]?\d+"
    bid_count = len(re.findall(bid_pattern, next_state.lower()))
    
    # Optimal range of interactive elements suggests good state
    if 3 <= bid_count <= 25:
        q_value += 0.1
    elif bid_count > 0:
        q_value += 0.05
    
    # Check for completion indicators in next state
    completion_signals = ['success', 'saved', 'created', 'added', 'sent', 
                          'confirmed', 'completed', 'done', 'finish', 
                          'message sent', 'event created', 'task added']
    next_lower = next_state.lower()
    
    for signal in completion_signals:
        if signal in next_lower:
            q_value += 0.35
            break
    
    # Check for error indicators (negative signal)
    error_signals = ['error', 'failed', 'invalid', 'not found', 'unable', 
                     'cannot', 'unavailable', '404', '500']
    for signal in error_signals:
        if signal in next_lower:
            q_value -= 0.2
            break
    
    # Evaluate action quality
    action_lower = action.lower()
    
    if 'click' in action_lower:
        # Clicks are generally productive
        if 'submit' in next_lower or 'save' in next_lower:
            q_value += 0.2
        else:
            q_value += 0.1
    elif 'fill' in action_lower:
        # Form filling is productive progress
        q_value += 0.15
    elif 'press' in action_lower:
        # Key presses can be productive
        q_value += 0.1
    elif 'noop' in action_lower:
        # No-ops are weakly negative (wasted step)
        q_value -= 0.1
    elif 'scroll' in action_lower:
        # Scrolling is neutral/weakly positive
        q_value += 0.05
    
    # Check if we're in a goal-appropriate app context
    app_contexts = ['todo', 'calendar', 'messenger', 'maps', 'editor', 'code']
    for context in app_contexts:
        if context in next_lower:
            q_value += 0.05
            break
    
    # Penalize excessive state changes (might indicate navigation away)
    if state_change > 500:
        q_value -= 0.1
    
    # Normalize Q-value to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)