def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Start with base Q-value of 0
    q_value = 0.0
    
    # Check for immediate success in next_state
    success_patterns = [
        'success', 'completed', 'done', 'saved', 'created',
        'sent', 'added', 'confirmed', 'submitted'
    ]
    
    next_state_lower = next_state.lower()
    for pattern in success_patterns:
        if pattern in next_state_lower:
            return 1.0
    
    # Analyze action type for productivity
    action_lower = action.lower()
    
    # Assign base score based on action productivity
    if 'fill' in action_lower:
        action_score = 0.5
    elif 'click' in action_lower:
        action_score = 0.35
    elif 'press' in action_lower:
        action_score = 0.25
    elif 'scroll' in action_lower:
        action_score = 0.15
    else:
        action_score = 0.1
    
    # Penalize noop actions
    if 'noop' in action_lower:
        action_score *= 0.3
    
    # Analyze state progress
    state_lower = state.lower()
    
    # Count task-related elements
    task_elements = ['input', 'form', 'button', 'submit', 'save', 'event', 'message', 'todo']
    state_element_count = sum(1 for elem in task_elements if elem in state_lower)
    next_element_count = sum(1 for elem in task_elements if elem in next_state_lower)
    
    # Progress bonus if we're making progress
    progress_bonus = 0.0
    if next_element_count > state_element_count:
        progress_bonus = 0.25
    elif next_element_count == state_element_count:
        progress_bonus = 0.1
    else:
        progress_bonus = 0.05
    
    # Check for bid changes (interactive elements)
    bids_state = len(re.findall(r"bid['\"]?\s*[:=]\s*\d+", state))
    bids_next = len(re.findall(r"bid['\"]?\s*[:=]\s*\d+", next_state))
    
    if bids_next > bids_state:
        progress_bonus += 0.1
    
    # Check for URL/page changes (indicates navigation progress)
    if 'url' in state_lower or 'href' in state_lower:
        if state_lower != next_state_lower:
            progress_bonus += 0.1
    
    # Combine action and progress scores
    q_value = action_score * 0.6 + progress_bonus * 0.4
    
    # Ensure Q-value is in [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value