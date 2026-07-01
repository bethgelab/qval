def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment.
    
    Heuristics for value estimation:
    1. Task completion indicators (high value if success detected)
    2. Error states (reduce value)
    3. Presence of interactive elements (ability to make progress)
    4. App context recognition (being in the right application)
    5. Navigation structure (being on a main/accessible page)
    """
    
    state_lower = state.lower()
    value = 0.05  # Base value for being in the environment
    
    # Check for success/completion indicators
    success_indicators = [
        'success', 'saved', 'created', 'added', 'sent', 'completed', 
        'confirmed', 'done', 'finished', 'task created', 'event added',
        'message sent', 'note saved', 'appointment', 'reminder set',
        'location found', 'code saved', 'file created'
    ]
    for indicator in success_indicators:
        if indicator in state_lower:
            value = 0.95
            break
    
    # Check for error states (penalty)
    error_indicators = [
        'error', 'failed', 'invalid', 'not found', 'unavailable', 
        'timeout', 'blocked', 'denied', 'permission', 'missing'
    ]
    error_count = sum(1 for ind in error_indicators if ind in state_lower)
    if error_count > 0:
        value = max(0.0, value - error_count * 0.25)
    
    # Check for interactive elements (ability to make progress)
    interactive_elements = [
        'button', 'input', 'click', 'submit', 'save', 'send', 
        'add', 'create', 'new', 'form', 'field', 'text', 'textarea',
        'select', 'option', 'checkbox', 'radio', 'link', 'href'
    ]
    element_count = sum(1 for elem in interactive_elements if elem in state_lower)
    
    # More elements = better ability to complete task
    element_bonus = min(element_count * 0.025, 0.35)
    value += element_bonus
    
    # Check for app-specific context
    app_contexts = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor', 'task', 'event', 'message', 'location', 'file']
    context_found = sum(1 for ctx in app_contexts if ctx in state_lower)
    
    # Having app context is good
    context_bonus = min(context_found * 0.04, 0.20)
    value += context_bonus
    
    # Check for navigation elements (being on a main/accessible page)
    nav_elements = ['home', 'menu', 'page', 'view', 'list', 'items', 'entries', 'dashboard', 'main', 'content']
    nav_count = sum(1 for nav in nav_elements if nav in state_lower)
    
    # Navigation elements suggest we're on a main page
    nav_bonus = min(nav_count * 0.015, 0.10)
    value += nav_bonus
    
    # Check for bid/element tags (indicates Set-of-Marks visualization, good for actionability)
    if 'bid' in state_lower or 'tag' in state_lower:
        value += 0.05
    
    # Check for scrollable content (indicates we can explore more)
    if 'scroll' in state_lower or 'overflow' in state_lower:
        value += 0.03
    
    # Check for loading/processing states (temporary, slightly reduce value)
    loading_indicators = ['loading', 'loading...', 'processing', 'wait', 'spinner']
    if any(ind in state_lower for ind in loading_indicators):
        value = max(0.0, value - 0.1)
    
    # Clamp value to valid range [0, 1]
    value = max(0.0, min(1.0, value))
    
    return value