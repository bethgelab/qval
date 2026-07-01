def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check if goal is already achieved (binary reward = 1.0)
    success_patterns = [
        'goal achieved', 'task complete', 'success', 'completed', 
        'done', 'verified', 'target state', 'objective met'
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            return 1.0
    
    # Identify which app we're working with
    app_context = 0.0
    todo_patterns = ['todo', 'task', 'list', 'add', 'item', 'checkbox']
    calendar_patterns = ['calendar', 'event', 'date', 'schedule', 'appointment']
    messenger_patterns = ['message', 'chat', 'send', 'conversation', 'inbox']
    maps_patterns = ['map', 'location', 'route', 'directions', 'address']
    code_patterns = ['code', 'editor', 'file', 'script', 'syntax']
    
    if any(p in state_lower for p in todo_patterns):
        app_context += 0.2
    if any(p in state_lower for p in calendar_patterns):
        app_context += 0.2
    if any(p in state_lower for p in messenger_patterns):
        app_context += 0.2
    if any(p in state_lower for p in maps_patterns):
        app_context += 0.2
    if any(p in state_lower for p in code_patterns):
        app_context += 0.2
    
    # Count interactive elements available (more = better for progress)
    bid_matches = re.findall(r"bid'\s*=\s*'(\d+)'", state)
    bid_count = len(bid_matches)
    interactive_bonus = min(bid_count * 0.03, 0.3)
    
    # Check for form input readiness (text fields, inputs)
    input_indicators = ['textbox', 'input', 'textarea', 'field', 'form']
    has_inputs = any(ind in state_lower for ind in input_indicators)
    input_bonus = 0.15 if has_inputs else 0.0
    
    # Check for button/submit availability
    button_indicators = ['button', 'submit', 'save', 'send', 'add', 'create']
    has_buttons = any(btn in state_lower for btn in button_indicators)
    button_bonus = 0.15 if has_buttons else 0.0
    
    # Check for progress indicators (things already done)
    progress_indicators = ['selected', 'filled', 'added', 'created', 'saved', 'checked', 'entered']
    progress_count = sum(1 for p in progress_indicators if p in state_lower)
    progress_bonus = min(progress_count * 0.1, 0.3)
    
    # Penalize for error states or blocked progress
    error_patterns = ['error', 'failed', 'invalid', 'not found', 'cannot', 'blocked', 'loading']
    error_count = sum(1 for e in error_patterns if e in state_lower)
    error_penalty = min(error_count * 0.15, 0.4)
    
    # Check if we're in a dead end (no interactive elements at all)
    dead_end_penalty = 0.0
    if bid_count == 0 and not has_inputs and not has_buttons:
        dead_end_penalty = 0.3
    
    # Base value with all components
    base_value = 0.25
    
    # Combine all factors
    estimated_value = (
        base_value +
        app_context +
        interactive_bonus +
        input_bonus +
        button_bonus +
        progress_bonus -
        error_penalty -
        dead_end_penalty
    )
    
    # Clamp to valid range
    return max(0.0, min(1.0, estimated_value))