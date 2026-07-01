def signal_function(state: str) -> float:
    """
    Estimates state-value for OpenApps environment based on state features.
    Returns value in [0, 1] representing estimated probability of task success.
    """
    import re
    
    state_lower = state.lower()
    
    # Check for goal completion signals
    completion_signals = ['completed', 'success', 'saved', 'created', 'added', 
                          'sent', 'done', 'finished', '✓', '✔', 'event created',
                          'message sent', 'task added', 'calendar updated',
                          'todo completed', 'meeting scheduled']
    
    for signal in completion_signals:
        if signal in state_lower:
            return 1.0
    
    # Check for error/failure signals
    error_signals = ['error', 'failed', 'invalid', 'denied', 'rejected', 
                     '×', '✗', 'problem', 'issue', 'unauthorized']
    
    for signal in error_signals:
        if signal in state_lower:
            return 0.05
    
    # Check for progress indicators
    progress_score = 0.0
    
    # Form field indicators (filled content suggests progress)
    field_patterns = [r'fill.*?value', r'placeholder.*?text', r'input.*?value',
                      r'form.*?filled', r'contenteditable.*?text']
    for pattern in field_patterns:
        if re.search(pattern, state_lower):
            progress_score += 0.1
    
    # Action button presence (relevant interactive elements)
    action_keywords = ['submit', 'save', 'send', 'add', 'create', 'confirm',
                       'done', 'next', 'finish', 'ok', 'apply', 'update']
    for keyword in action_keywords:
        if keyword in state_lower:
            progress_score += 0.05
            break
    
    # Application context indicators
    app_context = 0.0
    app_keywords = ['calendar', 'todo', 'messenger', 'maps', 'editor',
                    'event', 'task', 'message', 'location', 'code']
    for keyword in app_keywords:
        if keyword in state_lower:
            app_context += 0.1
            break
    
    # Step estimation (fewer steps remaining = higher value)
    # Try to detect step count in state
    step_match = re.search(r'step[:\s]+(\d+)', state_lower)
    if step_match:
        current_step = int(step_match.group(1))
        steps_remaining = max(0, 45 - current_step)
        step_factor = steps_remaining / 45.0
    else:
        step_factor = 0.5  # Default assumption
    
    # Bid/tag presence indicates interactive elements available
    bid_count = len(re.findall(r'bid["\']?\s*[:=]?\s*["\']?\d+', state_lower))
    interaction_factor = min(1.0, bid_count / 10.0)
    
    # Combine factors
    base_value = 0.1  # Minimum baseline value
    progress_contribution = min(0.3, progress_score)
    context_contribution = min(0.2, app_context)
    interaction_contribution = min(0.1, interaction_factor * 0.1)
    
    value = base_value + progress_contribution + context_contribution + interaction_contribution
    value = value * step_factor
    
    # Clamp to valid range
    return max(0.0, min(1.0, value))