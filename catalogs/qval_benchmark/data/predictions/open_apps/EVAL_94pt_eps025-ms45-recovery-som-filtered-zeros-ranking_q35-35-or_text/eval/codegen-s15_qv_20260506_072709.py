def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value starts at 0 (no completion)
    q_value = 0.0
    
    # Check if goal is achieved in next_state (immediate reward)
    goal_keywords = ['success', 'completed', 'done', 'added', 'sent', 'created', 'saved', 'posted', 'confirmed']
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    
    for keyword in goal_keywords:
        if keyword in next_state_lower:
            q_value = 1.0
            break
    
    # If not completed, estimate progress toward goal
    if q_value < 1.0:
        # Action quality scoring
        action_lower = action.lower()
        
        # Fill actions are typically closer to completion (form-based tasks)
        if 'fill' in action_lower:
            q_value += 0.15
        
        # Click on submit/save/confirm buttons
        if any(btn in action_lower for btn in ['submit', 'save', 'confirm', 'add', 'send', 'create']):
            q_value += 0.20
        
        # Navigation actions (moving toward goal)
        if 'click' in action_lower and any(nav in action_lower for nav in ['add', 'new', 'compose', 'create']):
            q_value += 0.10
        
        # Scroll to find more elements
        if 'scroll' in action_lower:
            q_value += 0.05
        
        # Noop - typically not productive unless waiting
        if 'noop' in action_lower:
            q_value += 0.02
        
        # Check state transitions for progress
        # If next_state has more interactive elements than state, likely making progress
        state_elements = len(re.findall(r'\[bid=\d+\]', state_lower))
        next_elements = len(re.findall(r'\[bid=\d+\]', next_state_lower))
        
        if next_elements > state_elements:
            q_value += 0.10
        elif next_elements == state_elements:
            q_value += 0.02
        
        # Check for error patterns (negative signal)
        error_patterns = ['error', 'failed', 'invalid', 'unable', 'cannot', 'not found', '404', '500']
        for error in error_patterns:
            if error in next_state_lower:
                q_value -= 0.15
                break
        
        # Check for form completion patterns
        form_indicators = ['form submitted', 'data saved', 'entry added', 'message sent']
        for indicator in form_indicators:
            if indicator in next_state_lower:
                q_value += 0.25
                break
        
        # Discount for steps taken (assume we're roughly at step 10-20 based on typical patterns)
        # Higher steps = lower Q-value (efficiency matters)
        step_penalty = 0.05
        if any(step in next_state_lower for step in ['step', 'progress']):
            step_penalty = 0.10
        
        q_value = max(0.0, min(1.0, q_value - step_penalty))
    
    # Ensure Q-value is in valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value