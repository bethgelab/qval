def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment.
    Returns a float in [0, 1] representing expected success probability.
    """
    import re
    
    value = 0.0
    state_lower = state.lower()
    
    # 1. Check for success/completion indicators (strong signal)
    success_patterns = ['success', 'completed', 'done', 'saved', 'sent', 'added', 'created', 'submitted']
    if any(pattern in state_lower for pattern in success_patterns):
        value += 0.5
    
    # 2. Identify current application context (moderate signal)
    app_indicators = {
        'todo': ['todo', 'task', 'checklist', 'checkbox'],
        'calendar': ['calendar', 'event', 'date', 'schedule', 'appointment'],
        'messenger': ['message', 'chat', 'conversation', 'send', 'inbox'],
        'maps': ['map', 'location', 'address', 'direction', 'place'],
        'code': ['code', 'editor', 'file', 'save', 'python', 'javascript']
    }
    
    app_found = False
    for app, indicators in app_indicators.items():
        if any(ind in state_lower for ind in indicators):
            value += 0.1
            app_found = True
            break
    
    if not app_found:
        value += 0.05  # Unknown app context
    
    # 3. Count interactive elements (bids indicate clickable/fillable elements)
    bid_matches = re.findall(r"bid['\"]?\s*[:=]\s*['\"]?(\d+)", state_lower)
    bid_count = len(bid_matches)
    if bid_count > 0:
        value += 0.05 * min(bid_count / 8, 1.0)
    
    # 4. Penalize error states (negative signal)
    error_patterns = ['error', 'failed', 'invalid', 'not found', 'cannot', 'unavailable', 'warning']
    error_count = sum(1 for pattern in error_patterns if pattern in state_lower)
    value -= 0.15 * min(error_count, 2)
    
    # 5. Check for navigation/progress indicators (moderate signal)
    progress_patterns = ['page', 'step', 'progress', 'loading', 'submit', 'next', 'continue']
    if any(pattern in state_lower for pattern in progress_patterns):
        value += 0.08
    
    # 6. Check for form elements (indicates active task engagement)
    form_elements = ['input', 'form', 'button', 'text', 'field', 'textarea']
    form_count = sum(1 for elem in form_elements if elem in state_lower)
    if form_count >= 2:
        value += 0.05 * min(form_count / 4, 1.0)
    
    # 7. Check for specific action completions (task-specific signals)
    action_completions = ['clicked', 'filled', 'entered', 'typed', 'pressed', 'opened', 'closed']
    if any(pattern in state_lower for pattern in action_completions):
        value += 0.05
    
    # Normalize value to [0, 1] range
    value = max(0.0, min(1.0, value))
    
    return value