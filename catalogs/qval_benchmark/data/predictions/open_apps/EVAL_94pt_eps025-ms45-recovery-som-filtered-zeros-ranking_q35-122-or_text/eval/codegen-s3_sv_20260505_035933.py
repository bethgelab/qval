def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment.
    
    Uses heuristic analysis of the state text to estimate how close
    we are to completing the task goal, considering:
    - Success/completion indicators
    - Progress markers (forms, buttons, pages)
    - Error/obstacle indicators
    - Available interactive elements
    """
    state_lower = state.lower()
    
    # Start with base value
    value = 0.0
    
    # Success indicators - strong positive signals
    success_patterns = [
        'success', 'completed', 'saved', 'created', 'sent', 'added',
        'event created', 'message sent', 'task completed', 'form submitted',
        'verification', 'confirmation', 'done', 'finished'
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            value += 0.25
    
    # Progress indicators - moderate positive signals
    progress_patterns = [
        'form', 'input', 'button', 'click', 'open', 'page', 'view',
        'list', 'item', 'select', 'option', 'field', 'text', 'textarea',
        'calendar', 'todo', 'message', 'map', 'code', 'editor'
    ]
    progress_count = sum(1 for p in progress_patterns if p in state_lower)
    value += min(progress_count * 0.03, 0.3)
    
    # Interactive elements - indicates we have actions available
    bid_count = state_lower.count('bid')
    if bid_count > 0:
        value += min(bid_count * 0.01, 0.2)
    
    # Error indicators - negative signals
    error_patterns = [
        'error', 'failed', 'invalid', 'not found', 'unavailable',
        'empty', 'required', 'missing', 'cannot', 'unable', 'timeout'
    ]
    error_count = sum(1 for p in error_patterns if p in state_lower)
    value -= error_count * 0.15
    
    # Warning indicators - mild negative signals
    warning_patterns = [
        'warning', 'alert', 'notice', 'please', 'must', 'should',
        'try again', 'retry', 'refresh'
    ]
    warning_count = sum(1 for p in warning_patterns if p in state_lower)
    value -= warning_count * 0.05
    
    # Check for navigation context - being in the right app matters
    app_context_patterns = [
        'todo', 'calendar', 'messenger', 'maps', 'code', 'editor'
    ]
    context_match = sum(1 for p in app_context_patterns if p in state_lower)
    if context_match > 0:
        value += context_match * 0.05
    
    # Check for goal-related keywords (task-specific progress)
    goal_patterns = [
        'title', 'description', 'date', 'time', 'location', 'recipient',
        'subject', 'body', 'content', 'note', 'comment', 'tag', 'category'
    ]
    goal_match = sum(1 for p in goal_patterns if p in state_lower)
    if goal_match > 0:
        value += goal_match * 0.02
    
    # Normalize and clamp to [0, 1] range
    # A state with strong success indicators should be near 1.0
    # A state with errors should be lower
    value = max(0.0, min(1.0, value))
    
    # Apply efficiency bonus - if we're making progress, value increases
    # This accounts for the step limit (45 steps) - earlier progress is better
    if value > 0.3:
        value = value * 1.1  # Boost promising states
    
    # Final normalization
    value = max(0.0, min(1.0, value))
    
    return float(value)