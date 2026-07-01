def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value estimate
    q_value = 0.0
    
    # Check for explicit success indicators in next_state
    success_patterns = [
        r'(?i)success',
        r'(?i)completed',
        r'(?i)task.*complete',
        r'(?i)goal.*reached',
        r'(?i)verified',
        r'(?i)saved',
        r'(?i)sent',
        r'(?i)created',
        r'(?i)added',
        r'(?i)updated'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state):
            return 1.0
    
    # Check for error/failure indicators that reduce Q-value
    error_patterns = [
        r'(?i)error',
        r'(?i)failed',
        r'(?i)invalid',
        r'(?i)missing.*required',
        r'(?i)not.*found',
        r'(?i)unavailable'
    ]
    
    error_penalty = 0.0
    for pattern in error_patterns:
        if re.search(pattern, next_state):
            error_penalty += 0.15
    
    q_value -= min(error_penalty, 0.5)
    
    # Assess action quality
    action_lower = action.lower()
    
    # Productive actions get positive bonus
    productive_actions = ['click', 'fill', 'type', 'press', 'submit', 'send', 'save']
    if any(act in action_lower for act in productive_actions):
        q_value += 0.15
    
    # No-op actions get negative penalty
    if 'noop' in action_lower:
        q_value -= 0.1
    
    # Assess state progress indicators
    # Count bid tags (interactive elements) in next_state
    bid_count = len(re.findall(r'bid=\d+', next_state))
    
    # More bids suggests more actionable state
    if bid_count >= 5:
        q_value += 0.1
    elif bid_count >= 2:
        q_value += 0.05
    
    # Check for form completion indicators
    form_patterns = [
        r'(?i)value=',
        r'(?i)checked',
        r'(?i)filled',
        r'(?i)selected'
    ]
    
    form_completion = 0
    for pattern in form_patterns:
        if re.search(pattern, next_state):
            form_completion += 0.05
    
    q_value += min(form_completion, 0.2)
    
    # Check if state changed meaningfully (progress vs stagnation)
    if len(next_state) > len(state) * 0.8:
        q_value += 0.05  # State evolved
    elif len(next_state) < len(state) * 0.3:
        q_value -= 0.1  # Possible page reset or error
    
    # Normalize to [0, 1] range
    q_value = max(0.0, min(1.0, q_value))
    
    # If we're in a clearly productive state with no errors, give moderate Q-value
    if bid_count >= 3 and error_penalty < 0.2 and q_value > 0.1:
        q_value = max(q_value, 0.3)
    
    return q_value