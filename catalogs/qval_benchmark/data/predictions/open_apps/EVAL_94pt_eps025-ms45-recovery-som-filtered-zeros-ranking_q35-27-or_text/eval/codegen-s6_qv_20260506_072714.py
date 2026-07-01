def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    action_lower = action.lower().strip()
    
    # Action type scoring - productive actions get higher base value
    action_score = 0.0
    if action_lower.startswith('fill('):
        action_score = 0.5
    elif action_lower.startswith('click('):
        action_score = 0.4
    elif action_lower.startswith('press('):
        action_score = 0.3
    elif action_lower.startswith('scroll('):
        action_score = 0.1
    elif action_lower.startswith('noop('):
        action_score = 0.0
    else:
        action_score = 0.2
    
    # Success indicators - strong signal of goal achievement
    success_indicators = ['completed', 'success', 'saved', 'created', 'sent', 'added', 
                         'done', 'finished', 'confirmed', 'updated', 'posted', 'shared',
                         'event created', 'task completed', 'message sent', 'calendar event',
                         'new event', 'task added', 'message delivered', 'checkmark', '✓', '✔']
    
    success_score = 0.95 if any(ind.lower() in next_state.lower() for ind in success_indicators) else 0.0
    
    # State change detection - meaningful changes suggest progress
    change_score = 0.3 if state != next_state else 0.0
    
    # Form completion indicators - suggests task is advancing
    form_indicators = ['submit', 'save', 'confirm', 'create', 'send', 'add', 'update', 'delete']
    form_score = 0.25 if any(ind.lower() in next_state.lower() for ind in form_indicators) else 0.0
    
    # Error detection - errors reduce expected return
    error_indicators = ['error', 'failed', 'invalid', 'required', 'missing', 'not found', 'failed to']
    error_score = 0.4 if any(ind.lower() in next_state.lower() for ind in error_indicators) else 0.0
    
    # Combine scores with appropriate weights
    q_value = action_score + success_score + change_score + form_score - error_score
    
    # Clamp to valid Q-value range [0, 1]
    return max(0.0, min(1.0, q_value))