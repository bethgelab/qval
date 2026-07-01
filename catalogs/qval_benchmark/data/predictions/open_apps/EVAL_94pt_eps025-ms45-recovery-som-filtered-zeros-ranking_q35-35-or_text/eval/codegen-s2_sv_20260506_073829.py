def signal_function(state: str) -> float:
    """
    Estimate state-value based on direct analysis of state representation.
    Returns value between 0.0 (poor) and 1.0 (goal achieved).
    """
    state_lower = state.lower()
    
    # Check if goal is already achieved
    goal_achieved_patterns = [
        'success', 'completed', 'task completed', 'goal achieved',
        'event created', 'message sent', 'todo added', 'item added',
        'saved', 'done', 'finished', 'task complete'
    ]
    for pattern in goal_achieved_patterns:
        if pattern in state_lower:
            return 1.0
    
    # Check for progress indicators based on app type
    progress_score = 0.0
    step_penalty = 0.0
    
    # Calendar app indicators
    calendar_progress = sum(1 for p in ['calendar', 'event', 'date', 'time', 'add event', 'save event'] 
                           if p in state_lower)
    if calendar_progress > 0:
        progress_score += min(0.3, calendar_progress * 0.05)
    
    # Todo app indicators
    todo_progress = sum(1 for p in ['todo', 'task', 'add task', 'check', 'complete', 'list'] 
                       if p in state_lower)
    if todo_progress > 0:
        progress_score += min(0.3, todo_progress * 0.05)
    
    # Messenger app indicators
    messenger_progress = sum(1 for p in ['message', 'chat', 'send', 'inbox', 'compose', 'reply'] 
                            if p in state_lower)
    if messenger_progress > 0:
        progress_score += min(0.3, messenger_progress * 0.05)
    
    # Maps app indicators
    maps_progress = sum(1 for p in ['map', 'location', 'directions', 'search', 'address'] 
                       if p in state_lower)
    if maps_progress > 0:
        progress_score += min(0.2, maps_progress * 0.05)
    
    # Code editor indicators
    code_progress = sum(1 for p in ['code', 'editor', 'save', 'run', 'compile'] 
                       if p in state_lower)
    if code_progress > 0:
        progress_score += min(0.2, code_progress * 0.05)
    
    # Check if we're on a navigation page (bad progress)
    navigation_patterns = ['home', 'menu', 'navigation', 'sidebar', 'main page']
    nav_penalty = sum(1 for p in navigation_patterns if p in state_lower)
    step_penalty += min(0.2, nav_penalty * 0.05)
    
    # Check for error states
    error_patterns = ['error', 'not found', 'invalid', 'failed', 'cannot', 'disabled']
    error_penalty = sum(1 for p in error_patterns if p in state_lower)
    step_penalty += min(0.3, error_penalty * 0.1)
    
    # Calculate final value
    base_value = progress_score - step_penalty
    
    # Ensure value is in valid range
    value = max(0.0, min(1.0, base_value))
    
    return value