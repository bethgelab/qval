def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment based on state text representation.
    Returns a float between 0.0 and 1.0 representing the estimated value.
    """
    state_lower = state.lower()
    
    # Start with base value for being in a valid state
    value = 0.15
    
    # Check for success indicators (highest priority - immediate return)
    success_keywords = [
        'created', 'added', 'sent', 'saved', 'completed', 
        'success', 'done', 'confirmed', 'submitted',
        'event created', 'message sent', 'task added',
        'calendar event', 'todo completed', 'note saved',
        'task created', 'appointment', 'invitation'
    ]
    
    success_count = sum(1 for kw in success_keywords if kw in state_lower)
    if success_count > 0:
        return 1.0  # Goal achieved
    
    # Check for error indicators (should lower value)
    error_keywords = [
        'error', 'failed', 'not found', 'invalid',
        'required', 'missing', 'empty', 'cannot',
        'unavailable', 'timeout', 'unauthorized'
    ]
    
    error_count = sum(1 for kw in error_keywords if kw in state_lower)
    value -= 0.08 * error_count
    
    # Check for form/field elements (indicates we can make progress)
    form_keywords = ['input', 'textbox', 'field', 'textarea', 'select', 'dropdown']
    form_count = sum(1 for kw in form_keywords if kw in state_lower)
    value += 0.02 * min(form_count, 6)
    
    # Check for interactive elements (bid tags suggest clickable items)
    bid_count = state_lower.count('bid')
    value += 0.006 * min(bid_count, 30)
    
    # Check for specific app contexts that might indicate progress
    app_contexts = {
        'calendar': 0.10,
        'todo': 0.08,
        'messenger': 0.07,
        'maps': 0.05,
        'editor': 0.07,
        'code': 0.07
    }
    
    for app, bonus in app_contexts.items():
        if app in state_lower:
            value += bonus
    
    # Check for navigation elements (indicates we're moving through the app)
    nav_keywords = ['home', 'dashboard', 'menu', 'link', 'page', 'navigate', 'tab']
    nav_count = sum(1 for kw in nav_keywords if kw in state_lower)
    value += 0.01 * nav_count
    
    # Check for completion indicators (checkboxes, checkmarks)
    completion_keywords = ['checked', 'checkmark', 'tick', 'done', 'complete']
    completion_count = sum(1 for kw in completion_keywords if kw in state_lower)
    value += 0.03 * completion_count
    
    # Check for action verbs that suggest progress
    action_keywords = ['click', 'press', 'fill', 'open', 'create', 'add', 'send', 'submit']
    action_count = sum(1 for kw in action_keywords if kw in state_lower)
    value += 0.008 * min(action_count, 8)
    
    # Check for title/header information indicating current page
    title_keywords = ['title', 'header', 'heading', 'h1', 'h2']
    title_count = sum(1 for kw in title_keywords if kw in state_lower)
    value += 0.01 * min(title_count, 3)
    
    # Normalize value to [0.0, 1.0] range
    value = max(0.0, min(1.0, value))
    
    return value