def signal_function(state: str) -> float:
    import re
    
    # Initialize value estimate
    value = 0.5  # Neutral starting point
    
    # Check if goal appears to be completed
    # Look for common goal completion indicators
    completion_keywords = [
        'saved', 'created', 'added', 'sent', 'confirmed',
        'success', 'complete', 'done', 'finished',
        'event created', 'message sent', 'todo added',
        'appointment', 'reminder', 'scheduled', 'submitted'
    ]
    
    for keyword in completion_keywords:
        if keyword.lower() in state.lower():
            value = min(1.0, value + 0.3)
            break
    
    # Check for error states
    error_keywords = [
        'error', 'failed', 'invalid', 'missing', 'not found',
        'cannot', 'unable', 'rejected', 'invalid input',
        'failed to', 'unexpected'
    ]
    
    for keyword in error_keywords:
        if keyword.lower() in state.lower():
            value = max(0.0, value - 0.4)
            break
    
    # Look for bid numbers (interactive elements)
    bid_pattern = r'\bid(\d+)'
    bids = re.findall(bid_pattern, state)
    
    if len(bids) > 0:
        # More bids might indicate more interactive elements
        # but too many could mean complexity or distraction
        num_bids = len(bids)
        if num_bids >= 15:
            value = max(0.0, value - 0.05)  # Too complex
        elif num_bids <= 3:
            value = min(1.0, value + 0.05)  # Simpler state
    
    # Look for form field indicators
    form_field_patterns = [
        r'input', r'textbox', r'textarea', r'select',
        r'checkbox', r'radio', r'button', r'link'
    ]
    form_fields = []
    for pattern in form_field_patterns:
        form_fields.extend(re.findall(pattern, state, re.IGNORECASE))
    
    if len(form_fields) > 0:
        # Filled forms indicate progress
        num_forms = len(set(form_fields))
        if num_forms >= 5:
            value = min(1.0, value + 0.1)
    
    # Look for navigation indicators (might indicate not at goal)
    nav_keywords = ['home', 'back', 'previous', 'page', 'navigate']
    for keyword in nav_keywords:
        if keyword.lower() in state.lower():
            value = max(0.0, value - 0.05)
    
    # Look for calendar-specific indicators
    calendar_keywords = ['calendar', 'event', 'date', 'time', 'appointment', 'schedule']
    if any(kw.lower() in state.lower() for kw in calendar_keywords):
        value = min(1.0, value + 0.05)
    
    # Look for message-specific indicators
    message_keywords = ['message', 'chat', 'send', 'recipient', 'conversation']
    if any(kw.lower() in state.lower() for kw in message_keywords):
        value = min(1.0, value + 0.05)
    
    # Look for todo-specific indicators
    todo_keywords = ['todo', 'task', 'list', 'complete', 'pending']
    if any(kw.lower() in state.lower() for kw in todo_keywords):
        value = min(1.0, value + 0.05)
    
    # Clamp value to [0, 1]
    value = max(0.0, min(1.0, value))
    
    return value