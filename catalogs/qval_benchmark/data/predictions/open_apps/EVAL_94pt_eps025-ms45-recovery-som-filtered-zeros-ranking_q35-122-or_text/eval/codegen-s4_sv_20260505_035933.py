def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment.
    
    Returns a float between 0 and 1 representing the estimated probability
    of achieving the goal from this state, accounting for remaining steps.
    """
    import re
    
    value = 0.0
    episode_length = 45
    
    # Extract current step from state
    step_match = re.search(r'(?:step|current_step)\s*[:=]\s*(\d+)', state, re.IGNORECASE)
    if step_match:
        current_step = int(step_match.group(1))
        steps_remaining = max(0, episode_length - current_step)
    else:
        steps_remaining = episode_length
    
    # Calculate base value from remaining steps (time pressure)
    # More steps = more opportunity to succeed
    time_factor = steps_remaining / episode_length
    base_value = 0.2 * time_factor
    value += base_value
    
    # Check for explicit success/completion indicators
    success_keywords = [
        'success', 'completed', 'saved', 'sent', 'added', 'created',
        'confirmed', 'done', 'task complete', 'goal achieved', 'finished'
    ]
    
    success_found = False
    for keyword in success_keywords:
        if keyword in state.lower():
            success_found = True
            value += 0.5
            break
    
    # Check for error/failure indicators
    error_keywords = [
        'error', 'failed', 'fail', 'invalid', 'not found', 'unable',
        'cannot', 'blocked', 'rejected', 'unavailable'
    ]
    
    error_count = 0
    for keyword in error_keywords:
        if keyword in state.lower():
            error_count += 1
    
    if error_count > 0:
        value -= min(error_count * 0.08, 0.25)
    
    # Check for interactive elements (bid tags indicate actionable items)
    bid_count = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?(\d+)", state))
    if bid_count > 0:
        value += min(bid_count * 0.02, 0.15)
    
    # Check for task-relevant context
    task_keywords = ['calendar', 'messenger', 'todo', 'maps', 'editor', 'code']
    task_found = any(kw in state.lower() for kw in task_keywords)
    if task_found:
        value += 0.05
    
    # Check for form/input elements indicating progress
    form_keywords = ['input', 'field', 'form', 'text', 'textarea', 'select']
    form_count = sum(1 for kw in form_keywords if kw in state.lower())
    if form_count > 0:
        value += min(form_count * 0.03, 0.12)
    
    # Check for navigation progress indicators
    nav_patterns = [r'page\s*\d+', r'step\s*\d+', r'progress\s*[:=]\s*\d+', r'(\d+)%']
    for pattern in nav_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            value += 0.05
            break
    
    # Normalize to [0, 1] range
    value = max(0.0, min(1.0, value))
    
    return value