def signal_function(state: str) -> float:
    """
    Estimate state-value based on accessibility tree text representation.
    
    Returns a float in [0.0, 1.0] representing estimated expected discounted
    cumulative reward from this state assuming optimal play.
    """
    import re
    
    # Initialize value components
    progress_score = 0.0
    goal_score = 0.0
    block_score = 0.0
    context_score = 0.0
    interaction_score = 0.0
    
    state_lower = state.lower()
    
    # Progress indicators - elements suggesting we're moving toward goal
    progress_keywords = ['button', 'submit', 'save', 'add', 'create', 
                         'send', 'confirm', 'finish', 'complete', 'next',
                         'forward', 'proceed', 'enter', 'ok', 'yes']
    for kw in progress_keywords:
        if kw in state_lower:
            progress_score += 0.05
    
    # Goal completion indicators - suggests task is already done
    goal_keywords = ['completed', 'done', 'success', 'achieved', 'finished',
                     'goal met', 'task done', 'verified', 'target reached']
    for kw in goal_keywords:
        if kw in state_lower:
            goal_score += 0.15
    
    # Blocker indicators - suggests obstacles or errors
    block_keywords = ['error', 'warning', 'invalid', 'failed', 'blocked',
                      'disabled', 'unavailable', 'missing', 'not found',
                      'cannot', 'unable', 'permission denied', 'locked']
    for kw in block_keywords:
        if kw in state_lower:
            block_score += 0.1
    
    # Context indicators - relevant app/page presence
    context_keywords = ['calendar', 'todo', 'messenger', 'map', 'maps',
                        'code editor', 'form', 'input', 'text area',
                        'textarea', 'dropdown', 'checkbox', 'radio']
    for kw in context_keywords:
        if kw in state_lower:
            context_score += 0.03
    
    # Interaction opportunity count - number of actionable elements
    bid_pattern = r'bid["\']?\s*["\']?\s*\d+'
    bid_count = len(re.findall(bid_pattern, state_lower))
    interaction_score = min(bid_count * 0.02, 0.4)
    
    # Calculate final value
    value = 0.0
    
    # If goal already achieved, full value
    if goal_score >= 0.3:
        value = 1.0
    # If blocked, low value
    elif block_score >= 0.25:
        value = max(0.0, 0.1 - block_score)
    # Combine positive signals
    else:
        raw_value = progress_score + goal_score + context_score + interaction_score
        # Normalize to reasonable range
        value = min(1.0, max(0.0, raw_value * 1.5))
    
    return round(value, 4)