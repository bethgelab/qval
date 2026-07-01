def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Check for goal achievement indicators
    success_keywords = ['completed', 'success', 'done', 'finished', 'created', 'added', 'sent', 'saved']
    if any(kw in next_state_lower for kw in success_keywords):
        return 0.95
    
    # Check for error indicators that reduce value
    error_keywords = ['error', 'failed', 'invalid', 'incorrect', 'failed']
    if any(kw in next_state_lower for kw in error_keywords):
        return 0.1
    
    # Base value from state analysis
    base_value = 0.0
    
    # Detect app type and task context
    app_indicators = {
        'calendar': ['event', 'meeting', 'schedule', 'date', 'time', 'calendar'],
        'todo': ['task', 'item', 'checkbox', 'todo', 'list'],
        'messenger': ['message', 'send', 'recipient', 'chat', 'conversation'],
        'maps': ['location', 'address', 'route', 'direction', 'map'],
        'code': ['file', 'edit', 'code', 'save', 'run', 'editor']
    }
    
    detected_app = None
    for app, keywords in app_indicators.items():
        if any(kw in state_lower for kw in keywords):
            detected_app = app
            break
    
    # Score form completion and interactive elements
    form_progress = 0.0
    if re.search(r'filled|value=|input.*text|textbox', state_lower):
        form_progress += 0.15
    if 'submit' in state_lower or 'save' in state_lower or 'send' in state_lower:
        form_progress += 0.2
    if 'button' in state_lower or 'click' in state_lower:
        form_progress += 0.1
    if 'bid=' in state_lower:
        form_progress += 0.05
    
    base_value = min(0.5, form_progress)
    
    # Analyze action productivity
    action_value = 0.0
    if 'fill(' in action_lower:
        action_value = 0.4
    elif 'click(' in action_lower:
        action_value = 0.3
    elif 'press(' in action_lower:
        action_value = 0.25
    elif 'scroll(' in action_lower:
        action_value = 0.1
    elif 'noop(' in action_lower:
        action_value = 0.0
    
    # Check if state changed meaningfully
    state_changed = state != next_state
    if state_changed:
        action_value += 0.15
    
    # Analyze next state for improvement
    next_form_progress = 0.0
    if re.search(r'filled|value=|input.*text|textbox', next_state_lower):
        next_form_progress += 0.15
    if 'submit' in next_state_lower or 'save' in next_state_lower or 'send' in next_state_lower:
        next_form_progress += 0.2
    if 'button' in next_state_lower or 'click' in next_state_lower:
        next_form_progress += 0.1
    if 'bid=' in next_state_lower:
        next_form_progress += 0.05
    
    # State improvement bonus
    if next_form_progress > form_progress:
        action_value += 0.2
    
    # Check for bid changes indicating navigation
    state_bids = len(re.findall(r'bid=\d+', state_lower))
    next_bids = len(re.findall(r'bid=\d+', next_state_lower))
    if next_bids > state_bids and state_changed:
        action_value += 0.1
    
    # Combine values
    combined_value = base_value + action_value
    
    # Estimate steps remaining (heuristic based on progress)
    estimated_steps_remaining = max(1, 20 - int(combined_value * 15))
    
    # Discount factor
    gamma = 0.99
    
    # Q-value = probability of success * discounted future value
    success_probability = min(1.0, combined_value)
    estimated_return = success_probability * (gamma ** estimated_steps_remaining)
    
    # Scale to reasonable Q-value range
    q_value = estimated_return
    
    return max(0.0, min(1.0, q_value))