def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for success/completion indicators
    success_patterns = [
        'success', 'completed', 'done', 'saved', 'sent', 'added', 
        'created', 'updated', 'confirmed', 'ok', 'message sent',
        'event created', 'task added', 'calendar updated', 'todo added',
        'saved successfully', 'message sent', 'event saved'
    ]
    
    is_complete = any(pattern in state_lower for pattern in success_patterns)
    if is_complete:
        return 0.95
    
    # Count interactive elements (bids)
    bid_count = len(re.findall(r'bid\d+', state_lower))
    
    # Estimate based on available actions
    if bid_count >= 15:
        action_score = 0.35
    elif bid_count >= 10:
        action_score = 0.25
    elif bid_count >= 5:
        action_score = 0.18
    elif bid_count >= 2:
        action_score = 0.10
    elif bid_count >= 1:
        action_score = 0.05
    else:
        action_score = 0.0
    
    # Check for relevant content/app indicators
    relevant_keywords = [
        'todo', 'calendar', 'messenger', 'maps', 'code', 'editor',
        'task', 'event', 'message', 'map', 'file', 'save', 'send',
        'add', 'create', 'new', 'form', 'input', 'title', 'description',
        'date', 'time', 'location', 'to', 'from', 'subject', 'body'
    ]
    
    relevance_score = 0.0
    if any(kw in state_lower for kw in relevant_keywords):
        relevance_score = 0.25
    
    # Check for error states
    error_patterns = [
        'error', 'failed', 'invalid', 'not found', '404', '500',
        'unavailable', 'timeout', 'cannot', 'unable', 'no such',
        'required', 'missing', 'empty', 'blank'
    ]
    
    has_error = any(pattern in state_lower for pattern in error_patterns)
    error_penalty = 0.35 if has_error else 0.0
    
    # Check for loading states (uncertainty)
    loading_patterns = ['loading', 'processing', 'please wait', 'wait', 'fetching']
    is_loading = any(pattern in state_lower for pattern in loading_patterns)
    loading_penalty = 0.15 if is_loading else 0.0
    
    # Check for navigation/progress indicators
    nav_keywords = ['home', 'back', 'next', 'previous', 'continue', 'submit',
                   'click', 'press', 'enter', 'return', 'go', 'open']
    
    has_navigation = any(kw in state_lower for kw in nav_keywords)
    nav_score = 0.10 if has_navigation else 0.0
    
    # Calculate base value
    base_value = action_score + relevance_score + nav_score - error_penalty - loading_penalty
    
    # Ensure value is in valid range
    return min(1.0, max(0.0, base_value))