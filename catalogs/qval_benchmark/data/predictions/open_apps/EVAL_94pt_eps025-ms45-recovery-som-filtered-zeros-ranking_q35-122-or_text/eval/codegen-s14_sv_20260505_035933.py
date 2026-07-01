def signal_function(state: str) -> float:
    import re
    
    # Check for explicit goal completion indicators
    success_patterns = [
        r'completed', r'success', r'saved', r'sent', r'added', 
        r'created', r'confirmed', r'done', r'finished', r'task completed',
        r'successfully', r'confirmed'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, state.lower()):
            return 1.0
    
    # Check for failure/timeout indicators
    failure_patterns = [
        r'failed', r'error', r'timeout', r'expired', r'cancelled',
        r'cannot', r'unable', r'invalid'
    ]
    
    for pattern in failure_patterns:
        if re.search(pattern, state.lower()):
            return 0.0
    
    # Extract remaining steps from state if available
    remaining_steps = 45
    step_match = re.search(r'step[:\s]+(\d+)', state.lower())
    remaining_match = re.search(r'remaining[:\s]+(\d+)', state.lower())
    
    if remaining_match:
        remaining_steps = int(remaining_match.group(1))
    elif step_match:
        current_step = int(step_match.group(1))
        remaining_steps = max(0, 45 - current_step)
    
    # Step factor: more remaining steps = higher success probability
    # But we also value efficiency, so there's diminishing returns for too many steps
    if remaining_steps >= 30:
        step_factor = 0.9
    elif remaining_steps >= 20:
        step_factor = 0.75
    elif remaining_steps >= 10:
        step_factor = 0.6
    elif remaining_steps >= 5:
        step_factor = 0.4
    elif remaining_steps >= 2:
        step_factor = 0.25
    else:
        step_factor = 0.1
    
    # Count actionable/interactive elements as progress indicators
    action_elements = len(re.findall(r'button|input|form|link|field|clickable|select', state.lower()))
    
    # Element factor: more relevant elements = more opportunity to progress
    if action_elements >= 10:
        element_factor = 1.0
    elif action_elements >= 6:
        element_factor = 0.8
    elif action_elements >= 3:
        element_factor = 0.6
    elif action_elements >= 1:
        element_factor = 0.4
    else:
        element_factor = 0.2
    
    # Check for app context detection (being on correct application)
    app_keywords = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor', 'app']
    app_detected = any(kw in state.lower() for kw in app_keywords)
    
    # Navigation factor: being on the right app/page
    nav_factor = 0.8 if app_detected else 0.5
    
    # Look for progress indicators in the state
    progress_patterns = [r'progress', r'loading', r'processing', r'working']
    progress_detected = any(re.search(p, state.lower()) for p in progress_patterns)
    progress_bonus = 0.1 if progress_detected else 0.0
    
    # Base probability of success from random play
    base_probability = 0.15
    
    # Combine factors with appropriate weights
    # Step factor is most important (time pressure), then elements (opportunity), then navigation (context)
    estimated_value = (
        base_probability +
        step_factor * 0.45 +
        element_factor * 0.30 +
        nav_factor * 0.20 +
        progress_bonus
    )
    
    # Ensure value is in valid range [0.0, 1.0]
    return min(1.0, max(0.0, estimated_value))