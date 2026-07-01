def signal_function(state: str) -> float:
    import re
    
    value = 0.0
    state_lower = state.lower()
    
    # Check for task completion indicators (strong positive signal)
    completion_patterns = [
        r'success', r'completed', r'created', r'added', r'sent', r'saved',
        r'event created', r'message sent', r'task added', r'goal achieved',
        r'done', r'finished', r'submitted', r'updated', r'changed'
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, state_lower):
            value = max(value, 0.85)
    
    # Check for active progress indicators (medium positive signal)
    progress_patterns = [
        r'in progress', r'draft', r'editing', r'filling', r'entering',
        r'typing', r'selecting', r'creating', r'sending', r'adding'
    ]
    
    for pattern in progress_patterns:
        if re.search(pattern, state_lower):
            value = max(value, 0.5)
    
    # Check for form/input elements indicating task-relevant UI is present
    ui_element_patterns = [
        r'input', r'textarea', r'button', r'form', r'field', r'submit',
        r'bid', r'click', r'action', r'link'
    ]
    
    ui_count = sum(1 for pattern in ui_element_patterns if re.search(pattern, state_lower))
    if ui_count > 0:
        ui_bonus = min(0.3, 0.03 * ui_count)
        value = max(value, 0.3 + ui_bonus)
    
    # Check for error/failure indicators (negative signal)
    error_patterns = [
        r'error', r'failed', r'invalid', r'wrong', r'incorrect',
        r'not found', r'cannot', r'unable', r'permission denied'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, state_lower):
            value = min(value, 0.2)
            break
    
    # Check for step count information if available
    step_matches = re.findall(r'step[:\s]*(\d+)|step (\d+)', state_lower)
    if step_matches:
        for match in step_matches:
            step_str = match[0] or match[1]
            if step_str:
                try:
                    step = int(step_str)
                    remaining = 45 - step
                    if remaining > 0:
                        step_bonus = (remaining / 45) * 0.15
                        value = min(1.0, value + step_bonus)
                except (ValueError, TypeError):
                    pass
                break
    
    # Check for goal-related keywords in state
    goal_keywords = ['goal', 'target', 'objective', 'task', 'todo', 'calendar',
                     'event', 'message', 'map', 'editor', 'code']
    
    goal_count = sum(1 for kw in goal_keywords if kw in state_lower)
    if goal_count > 0:
        goal_bonus = min(0.2, 0.05 * goal_count)
        value = max(value, 0.25 + goal_bonus)
    
    # Ensure value stays in valid range [0.0, 1.0]
    return max(0.0, min(1.0, value))