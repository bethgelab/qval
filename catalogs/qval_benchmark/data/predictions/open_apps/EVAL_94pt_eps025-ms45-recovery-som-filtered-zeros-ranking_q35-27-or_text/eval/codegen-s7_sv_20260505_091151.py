def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Success indicators - strong signals that goal is achieved
    success_patterns = [
        'created', 'added', 'sent', 'saved', 'completed', 'done',
        'success', 'event created', 'task added', 'message sent',
        'item created', 'appointment', 'scheduled', 'successfully',
        'confirm', 'confirmed', '✓', '✔', 'check', 'finished'
    ]
    
    # Error indicators - signals of problems that reduce value
    error_patterns = [
        'error', 'failed', 'invalid', 'cannot', 'unable', 'denied',
        '×', '✗', 'failed to', 'could not', 'wrong', 'incorrect'
    ]
    
    # Progress indicators - signals of being on the right track
    progress_patterns = [
        'input', 'form', 'submit', 'button', 'calendar', 'event',
        'message', 'todo', 'task', 'new', 'create', 'add', 'send',
        'save', 'next', 'continue', 'finish', 'edit', 'update'
    ]
    
    # Interactive element indicators - agent can take action
    interactive_patterns = [
        'bid', 'click', 'button', 'link', 'input', 'textbox',
        'field', 'area', 'element', 'item'
    ]
    
    # App-specific goal states
    app_patterns = [
        'calendar', 'todo', 'messenger', 'maps', 'editor',
        'event', 'task', 'message', 'location', 'code'
    ]
    
    base_score = 0.5  # Start with neutral value
    
    # Check for success indicators (highest weight - goal achieved)
    for pattern in success_patterns:
        if pattern in state_lower:
            base_score = max(base_score, 0.95)
            break
    
    # Check for error indicators (penalty - reduces value)
    for pattern in error_patterns:
        if pattern in state_lower:
            base_score = min(base_score, 0.15)
            break
    
    # Check for progress indicators (moderate boost)
    progress_count = sum(1 for pattern in progress_patterns if pattern in state_lower)
    if progress_count > 0:
        progress_bonus = min(progress_count * 0.04, 0.25)
        base_score = min(base_score + progress_bonus, 0.85)
    
    # Check for interactive elements (can take action)
    interactive_count = sum(1 for pattern in interactive_patterns if pattern in state_lower)
    if interactive_count > 0:
        interactive_bonus = min(interactive_count * 0.015, 0.1)
        base_score = min(base_score + interactive_bonus, 0.95)
    
    # Check for app-specific patterns (context relevance)
    app_count = sum(1 for pattern in app_patterns if pattern in state_lower)
    if app_count > 0:
        app_bonus = min(app_count * 0.03, 0.15)
        base_score = min(base_score + app_bonus, 0.95)
    
    # Ensure score is in valid range [0.0, 1.0]
    return max(0.0, min(1.0, base_score))