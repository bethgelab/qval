def signal_function(state: str) -> float:
    """
    Estimates state-value for OpenApps environment based on state features.
    Returns a float in [0, 1] representing expected discounted cumulative reward.
    """
    import re
    
    # Normalize state for analysis
    state_lower = state.lower() if state else ""
    
    # Initialize base value
    base_value = 0.3  # Default baseline
    
    # Feature scoring
    score = 0.0
    penalty = 0.0
    
    # Check for goal achievement indicators (highest priority)
    success_patterns = [
        'success', 'completed', 'saved', 'created', 'sent', 
        'done', 'finished', 'event added', 'message sent',
        'task added', '✓', '✔', '✅'
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            score += 0.4
            break
    
    # Check for goal-related action buttons
    goal_buttons = [
        'save', 'submit', 'send', 'add', 'create', 'confirm',
        'done', 'finish', 'complete'
    ]
    button_count = 0
    for btn in goal_buttons:
        if btn in state_lower:
            button_count += 1
    if button_count > 0:
        score += min(0.25, button_count * 0.08)
    
    # Check for form completion (filled fields vs empty)
    filled_indicators = [
        'value=', 'filled', 'completed', 'entered', 'typed',
        'contenteditable', 'placeholder'
    ]
    empty_indicators = [
        'empty', 'required', 'not filled', 'no value', 'null'
    ]
    
    filled_count = sum(1 for p in filled_indicators if p in state_lower)
    empty_count = sum(1 for p in empty_indicators if p in state_lower)
    
    if filled_count > empty_count:
        score += 0.15
    elif empty_count > filled_count:
        penalty += 0.1
    
    # Check for error/warning states
    error_patterns = [
        'error', 'warning', 'failed', 'invalid', 'cannot',
        'denied', 'blocked', 'access denied', 'permission'
    ]
    error_count = sum(1 for p in error_patterns if p in state_lower)
    if error_count > 0:
        penalty += min(0.3, error_count * 0.15)
    
    # Check for navigation/progress indicators
    nav_patterns = [
        'next', 'continue', 'forward', 'step', 'page',
        'breadcrumb', 'menu', 'sidebar'
    ]
    nav_count = sum(1 for p in nav_patterns if p in state_lower)
    if nav_count > 0:
        score += min(0.1, nav_count * 0.03)
    
    # Check for interactive elements (more options = more flexibility)
    bid_pattern = re.compile(r'bid[=:]\s*\d+', re.IGNORECASE)
    bid_matches = bid_pattern.findall(state)
    if len(bid_matches) >= 10:
        score += 0.05
    elif len(bid_matches) >= 5:
        score += 0.02
    
    # Check for app-specific goal indicators
    app_goals = [
        'calendar', 'event', 'meeting', 'appointment',
        'todo', 'task', 'item', 'list',
        'message', 'chat', 'conversation', 'sender',
        'map', 'location', 'route', 'direction',
        'code', 'editor', 'file', 'save'
    ]
    app_match = sum(1 for p in app_goals if p in state_lower)
    if app_match >= 2:
        score += 0.1
    
    # Calculate final value
    value = base_value + score - penalty
    
    # Clamp to valid range
    value = max(0.0, min(1.0, value))
    
    # Adjust for estimated remaining steps (heuristic)
    # More filled forms/buttons suggests closer to completion
    progress_ratio = (filled_count + button_count) / max(1, filled_count + empty_count + 1)
    step_bonus = progress_ratio * 0.15
    
    value = min(1.0, value + step_bonus)
    
    return round(value, 4)