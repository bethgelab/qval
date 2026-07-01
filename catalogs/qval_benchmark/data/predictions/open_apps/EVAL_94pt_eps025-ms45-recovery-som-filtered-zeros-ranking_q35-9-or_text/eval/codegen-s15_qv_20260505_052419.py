def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on direct analysis of state representations.
    Returns a heuristic Q-value between 0 and 1 based on progress indicators.
    """
    import re
    
    base_q = 0.0
    action_lower = action.lower()
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Check for goal completion indicators in next state
    goal_keywords = ['goal', 'complete', 'success', 'done', 'verified', 'achieved', 'target', 'final']
    goal_complete = any(kw in next_state_lower for kw in goal_keywords)
    
    # Check for negative indicators (errors, warnings, failed states)
    error_keywords = ['error', 'fail', 'invalid', 'broken', 'missing', 'not found', 'unavailable']
    error_present = any(kw in next_state_lower for kw in error_keywords)
    
    # Check if action appears valid
    valid_action_types = ['click', 'fill', 'press', 'noop', 'scroll']
    action_valid = any(at in action_lower for at in valid_action_types)
    
    # Check for bid numbers (interactive elements)
    bid_count_state = len(re.findall(r'bid\s*\d+', state))
    bid_count_next = len(re.findall(r'bid\s*\d+', next_state))
    
    # Check for form-related progress
    form_keywords = ['form', 'input', 'text', 'field', 'email', 'phone', 'date', 'time']
    form_count_state = sum(1 for kw in form_keywords if kw in state_lower)
    form_count_next = sum(1 for kw in form_keywords if kw in next_state_lower)
    
    # Check for task-specific keywords
    task_keywords = ['todo', 'calendar', 'event', 'message', 'map', 'code', 'edit', 'save', 'submit']
    task_count_state = sum(1 for kw in task_keywords if kw in state_lower)
    task_count_next = sum(1 for kw in task_keywords if kw in next_state_lower)
    
    # Calculate progress signals
    if goal_complete:
        base_q = 0.95
    elif error_present:
        base_q = 0.05
    elif task_count_next > task_count_state:
        base_q = 0.4 + (task_count_next - task_count_state) * 0.1
    elif form_count_next > form_count_state:
        base_q = 0.35 + (form_count_next - form_count_state) * 0.08
    elif bid_count_next > bid_count_state:
        base_q = 0.25 + (bid_count_next - bid_count_state) * 0.05
    elif bid_count_next < bid_count_state:
        base_q = 0.15
    else:
        base_q = 0.1
    
    # Adjust based on action validity
    if action_valid:
        base_q = min(1.0, base_q * 1.1)
    else:
        base_q = max(0.0, base_q * 0.8)
    
    # Ensure reasonable range
    return round(max(0.0, min(1.0, base_q)), 4)