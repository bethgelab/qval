def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    q_value = 0.0
    
    # Check for task completion indicators in next_state
    completion_indicators = [
        r'saved', r'success', r'completed', r'done',
        r'added', r'sent', r'created', r'submitted',
        r'scheduled', r'message sent', r'event created',
        r'task added', r'event added', r'message sent'
    ]
    
    for indicator in completion_indicators:
        if re.search(indicator, next_state, re.IGNORECASE):
            q_value = 1.0
            return q_value
    
    # Evaluate action quality based on action type
    action_lower = action.lower()
    
    if 'fill' in action_lower:
        # Form filling is productive for task completion
        q_value += 0.35
    elif 'click' in action_lower:
        # Navigation clicks can be productive
        q_value += 0.25
    elif 'press' in action_lower:
        # Key presses vary in utility (enter, tab, etc.)
        q_value += 0.20
    elif 'scroll' in action_lower:
        # Scrolling is preparatory work
        q_value += 0.10
    elif 'noop' in action_lower:
        # No-op provides no progress
        q_value += 0.00
    
    # Check if next_state shows more progress than state
    # Count interactive element references (bids)
    state_bids = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?(\d+)", state))
    next_bids = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?(\d+)", next_state))
    
    if next_bids > state_bids:
        q_value += 0.08
    elif next_bids < state_bids * 0.7 and state_bids > 5:
        # Significant loss of elements suggests regression
        q_value -= 0.08
    
    # Check for form field indicators in next_state
    form_patterns = [r'input', r'textarea', r'select', r'form', r'checkbox', r'radio']
    form_count = sum(1 for p in form_patterns if re.search(p, next_state, re.IGNORECASE))
    q_value += min(0.15, form_count * 0.03)
    
    # Check for confirmation/feedback messages
    feedback_patterns = [r'thank you', r'confirmation', r'notice', r'alert', r'info']
    for pattern in feedback_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.10
            break
    
    # Penalize actions that seem to navigate away from task context
    if re.search(r'logout|sign out|exit|close', next_state, re.IGNORECASE):
        q_value -= 0.15
    
    # Ensure Q-value stays in valid range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value