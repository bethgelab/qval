def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    # Parse action type and parameters
    action_lower = action.lower()
    action_type = None
    action_target = None
    action_value = None
    
    if action_lower.startswith('click'):
        action_type = 'click'
        match = re.search(r"click\(['\"](\d+)['\"]", action)
        if match:
            action_target = match.group(1)
    elif action_lower.startswith('fill'):
        action_type = 'fill'
        match = re.search(r"fill\(['\"](\d+)['\"]", action)
        if match:
            action_target = match.group(1)
        match_val = re.search(r"fill\(['\"]\d+['\"]\s*,\s*['\"](.+?)['\"]", action)
        if match_val:
            action_value = match_val.group(1)
    elif action_lower.startswith('press'):
        action_type = 'press'
    elif action_lower.startswith('noop'):
        action_type = 'noop'
    elif action_lower.startswith('scroll'):
        action_type = 'scroll'
    
    # Analyze state changes
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Check if state changed significantly
    state_changed = state != next_state
    
    # Count interactive elements (bids) in state
    bid_pattern = re.compile(r'\bid(\d+)\b')
    state_bids = len(bid_pattern.findall(state))
    next_state_bids = len(bid_pattern.findall(next_state))
    
    # Goal-related keywords that suggest task completion
    goal_keywords = [
        'saved', 'created', 'added', 'completed', 'sent', 'done',
        'success', 'event', 'task', 'message', 'submitted', 'updated',
        'calendar', 'todo', 'messenger', 'maps', 'editor'
    ]
    
    # Check for success indicators in next state
    success_score = 0.0
    for keyword in goal_keywords:
        if keyword in next_state_lower:
            success_score += 0.1
    
    # Check for error indicators
    error_keywords = ['error', 'failed', 'invalid', 'required', 'missing', 'denied']
    error_score = 0.0
    for keyword in error_keywords:
        if keyword in next_state_lower:
            error_score += 0.15
    
    # Action quality scoring
    action_quality = 0.0
    
    if action_type == 'fill' and action_value:
        # Filling forms is generally productive
        action_quality = 0.3
        if len(action_value) > 0 and len(action_value) < 200:
            action_quality += 0.1
    elif action_type == 'click':
        # Clicking is productive
        action_quality = 0.25
    elif action_type == 'press':
        # Pressing keys can be productive (e.g., Enter to submit)
        action_quality = 0.2
    elif action_type == 'noop':
        # Noop is generally not productive
        action_quality = -0.1
    elif action_type == 'scroll':
        # Scrolling has moderate value
        action_quality = 0.05
    
    # State transition quality
    transition_quality = 0.0
    if state_changed:
        # State changed, which is generally good
        transition_quality = 0.2
        # Check if new content appeared
        new_words = set(next_state_lower.split()) - set(state_lower.split())
        if len(new_words) > 5:
            transition_quality += 0.1
    else:
        # No state change - could be bad unless action was noop
        if action_type != 'noop':
            transition_quality = -0.15
    
    # Bid availability score
    bid_score = 0.0
    if next_state_bids > 0:
        bid_score = min(0.15, next_state_bids * 0.005)
    
    # Combine scores
    base_q = action_quality + transition_quality + bid_score
    base_q += success_score * 2.0  # Success indicators are very important
    base_q -= error_score * 2.0    # Errors are very bad
    
    # Cap the Q-value between 0 and 1
    q_value = max(0.0, min(1.0, base_q))
    
    # If we detect strong success signals, boost the Q-value
    if success_score >= 0.3:
        q_value = max(q_value, 0.85)
    
    # If we detect errors, reduce the Q-value
    if error_score >= 0.3:
        q_value = min(q_value, 0.15)
    
    return q_value