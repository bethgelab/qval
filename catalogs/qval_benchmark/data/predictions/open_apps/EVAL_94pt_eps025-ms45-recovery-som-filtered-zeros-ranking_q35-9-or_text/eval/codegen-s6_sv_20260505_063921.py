def signal_function(state: str) -> float:
    import re
    import math
    
    # Normalize state text for analysis
    state_lower = state.lower()
    
    # Initialize value estimate
    value = 0.0
    
    # Check for goal completion indicators
    goal_indicators = ['goal achieved', 'success', 'completed', 'saved', 'created', 
                       'added', 'sent', 'message sent', 'event created', 'task done',
                       'notification', 'confirmation', 'done', 'finished']
    for indicator in goal_indicators:
        if indicator in state_lower:
            value = max(value, 0.95)
            break
    
    # Check for error indicators (penalize)
    error_indicators = ['error', 'failed', 'invalid', 'wrong', 'incorrect', 
                        'not found', 'missing', 'unable', 'blocked', 'timeout']
    error_count = 0
    for indicator in error_indicators:
        if indicator in state_lower:
            error_count += 1
    if error_count > 0:
        value = max(0.0, value - 0.3 * error_count)
    
    # Check for progress indicators
    progress_indicators = ['page', 'view', 'tab', 'form', 'field', 'input', 
                           'button', 'link', 'checkbox', 'radio', 'select', 'textarea']
    progress_count = 0
    for indicator in progress_indicators:
        if indicator in state_lower:
            progress_count += 1
    # More interactive elements suggest more context/progress
    if progress_count > 5:
        value = min(1.0, value + 0.1)
    elif progress_count > 0:
        value = min(1.0, value + 0.05)
    
    # Check for bid numbers (indicates interactive elements)
    bid_pattern = r'\bid\s*=\s*\d+'
    bid_matches = re.findall(bid_pattern, state)
    bid_count = len(bid_matches)
    # More bids = more navigation/interaction available
    if bid_count > 10:
        value = min(1.0, value + 0.05)
    elif bid_count > 5:
        value = min(1.0, value + 0.03)
    elif bid_count > 0:
        value = min(1.0, value + 0.02)
    
    # Check for form completion patterns
    form_fields = re.findall(r'input.*?name.*?=', state_lower)
    filled_fields = re.findall(r'input.*?value.*?=.*?[\w\s]+', state_lower)
    field_ratio = len(filled_fields) / max(len(form_fields), 1)
    value = value * (0.5 + 0.5 * field_ratio)
    
    # Check for navigation context
    app_indicators = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor', 
                      'app', 'page', 'view', 'screen']
    app_count = sum(1 for app in app_indicators if app in state_lower)
    if app_count > 0:
        value = min(1.0, value + 0.05)
    
    # Check for step budget (assume state contains step info)
    step_pattern = r'step.*?(\d+)'
    step_matches = re.findall(step_pattern, state)
    if step_matches:
        try:
            current_step = int(step_matches[0])
            remaining = 45 - current_step
            if remaining > 30:
                value = min(1.0, value + 0.05)
            elif remaining > 15:
                value = min(1.0, value + 0.03)
            elif remaining > 5:
                value = min(1.0, value + 0.02)
        except:
            pass
    
    # Check for heading levels (indicates page structure)
    heading_pattern = r'h[1-6]'
    heading_count = len(re.findall(heading_pattern, state_lower))
    if heading_count > 0:
        value = min(1.0, value + 0.02)
    
    # Check for list items (indicates task/list completion)
    list_pattern = r'(li|ul|ol).*?(?:checked|selected|active)'
    list_matches = re.findall(list_pattern, state_lower)
    if list_matches:
        value = min(1.0, value + 0.03)
    
    # Check for modal/dialog state (might indicate goal proximity)
    modal_indicators = ['modal', 'dialog', 'popup', 'overlay', 'confirm']
    modal_count = sum(1 for indicator in modal_indicators if indicator in state_lower)
    if modal_count > 0:
        value = min(1.0, value + 0.05)
    
    # Ensure value is in valid range
    value = max(0.0, min(1.0, value))
    
    return value