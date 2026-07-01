def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Normalize strings for analysis
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Start with neutral baseline
    q_value = 0.25
    
    # Check for success/completion indicators in next state
    success_keywords = [
        'success', 'completed', 'saved', 'added', 'sent', 'created',
        'done', 'confirmed', 'submitted', 'task added', 'event added',
        'message sent', 'note created', 'calendar event', 'todo added',
        'event created', 'message delivered', 'note saved'
    ]
    
    success_count = sum(1 for kw in success_keywords if kw in next_state_lower)
    q_value += success_count * 0.12
    
    # Check for failure/error indicators in next state
    fail_keywords = [
        'error', 'fail', 'invalid', 'not found', '404', '500',
        'cannot', 'unable', 'failed', 'incorrect', 'wrong',
        'empty', 'required', 'missing', 'unavailable'
    ]
    
    fail_count = sum(1 for kw in fail_keywords if kw in next_state_lower)
    q_value -= fail_count * 0.15
    
    # Validate action is a proper BrowserGym primitive
    valid_action_patterns = [
        r"click\s*\(",
        r"fill\s*\(",
        r"press\s*\(",
        r"noop\s*\(",
        r"scroll\s*\("
    ]
    
    action_valid = any(re.search(pat, action_lower) for pat in valid_action_patterns)
    if action_valid:
        q_value += 0.08
    
    # Check if action targets a bid (interactive element)
    bid_targeted = bool(re.search(r"['\"]?\d+['\"]?\s*\)", action_lower))
    if bid_targeted:
        q_value += 0.05
    
    # Count interactive elements as progress indicator
    bid_count = len(re.findall(r"\b\d+\b", state_lower))
    
    # Fewer bids remaining suggests we're closer to completion
    if bid_count < 10:
        q_value += 0.1
    elif bid_count < 20:
        q_value += 0.05
    
    # Check if state shows task context (which app we're in)
    app_contexts = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor']
    context_found = sum(1 for ctx in app_contexts if ctx in state_lower)
    if context_found > 0:
        q_value += 0.03
    
    # Check if next state shows progress (more content than before)
    next_content_len = len(next_state_lower)
    state_content_len = len(state_lower)
    if next_content_len > state_content_len:
        q_value += 0.02
    
    # Penalize if action seems to navigate away from current context
    if 'back' in action_lower or 'previous' in action_lower:
        q_value -= 0.05
    
    # Clamp value to reasonable Q-value range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value