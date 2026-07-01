def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Priority 1: Check for goal achievement indicators
    success_patterns = [
        'success', 'completed', 'created', 'saved', 'added', 'sent',
        'updated', 'done', 'finished', 'event created', 'message sent',
        'task added', 'item saved', 'scheduled', 'confirmed'
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            return 1.0
    
    # Priority 2: Check for critical errors or failures
    error_patterns = [
        'error', 'failed', 'invalid', 'denied', 'unauthorized',
        'not found', '404', '500', 'exception', 'timeout'
    ]
    for pattern in error_patterns:
        if pattern in state_lower:
            return 0.05
    
    # Priority 3: Assess application context (being on relevant app page)
    app_context_score = 0.0
    app_indicators = ['todo', 'calendar', 'messenger', 'maps', 'editor', 'app']
    app_count = sum(1 for app in app_indicators if app in state_lower)
    app_context_score = min(app_count * 0.15, 0.45)
    
    # Priority 4: Interactive elements availability (buttons, inputs, forms)
    interactive_score = 0.0
    if 'button' in state_lower:
        interactive_score += 0.15
    if 'submit' in state_lower:
        interactive_score += 0.15
    if 'input' in state_lower:
        interactive_score += 0.1
    if 'form' in state_lower:
        interactive_score += 0.1
    if 'bid' in state_lower:
        interactive_score += 0.1
    interactive_score = min(interactive_score, 0.5)
    
    # Priority 5: Form completion progress indicators
    form_score = 0.0
    if 'filled' in state_lower:
        form_score += 0.15
    if 'value' in state_lower:
        form_score += 0.1
    if 'content' in state_lower:
        form_score += 0.05
    if 'field' in state_lower:
        form_score += 0.05
    form_score = min(form_score, 0.35)
    
    # Priority 6: Navigation and page state
    nav_score = 0.0
    if 'page' in state_lower:
        nav_score += 0.05
    if 'link' in state_lower:
        nav_score += 0.05
    if 'open' in state_lower:
        nav_score += 0.05
    nav_score = min(nav_score, 0.15)
    
    # Priority 7: Task-specific progress indicators
    task_score = 0.0
    task_indicators = ['event', 'message', 'task', 'item', 'note', 'entry']
    task_count = sum(1 for t in task_indicators if t in state_lower)
    task_score = min(task_count * 0.08, 0.24)
    
    # Combine all scores
    value = app_context_score + interactive_score + form_score + nav_score + task_score
    
    # Ensure value is in valid range [0.0, 1.0]
    return max(0.0, min(1.0, value))