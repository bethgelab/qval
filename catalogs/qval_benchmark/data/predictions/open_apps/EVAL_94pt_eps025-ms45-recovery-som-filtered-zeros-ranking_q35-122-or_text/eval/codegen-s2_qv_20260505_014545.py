def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Completion indicators - keywords suggesting task progress or success
    completion_keywords = [
        'success', 'complete', 'done', 'saved', 'created', 'added', 
        'sent', 'event', 'message', 'todo', 'calendar', 'task',
        'submitted', 'confirmed', 'updated', 'opened', 'loaded'
    ]
    
    # Count completion-related signals in next state
    completion_score = 0
    for kw in completion_keywords:
        if kw.lower() in next_state.lower():
            completion_score += 0.15
    
    # Cap completion score contribution
    completion_score = min(completion_score, 0.5)
    
    # Analyze action type - meaningful actions contribute more
    action_lower = action.lower()
    
    # Check if action is productive (not noop or pure navigation)
    is_click = 'click' in action_lower and 'bid' in action_lower
    is_fill = 'fill' in action_lower and 'bid' in action_lower
    is_press = 'press' in action_lower and 'bid' in action_lower
    is_noop = action_lower.startswith('noop')
    is_scroll = 'scroll' in action_lower
    
    # Productive action bonus
    action_bonus = 0.0
    if is_click or is_fill or is_press:
        action_bonus = 0.25
    elif is_scroll:
        action_bonus = 0.1  # Scrolling can be productive for navigation
    elif is_noop:
        action_bonus = 0.0  # Noop provides no progress
    
    # Analyze state progression via bid count (interactive elements)
    # More bids typically means more task-relevant elements visible
    bid_pattern = r"bid['\"]?\s*[:=]\s*['\"]?(\d+)"
    
    state_bids = len(re.findall(bid_pattern, state))
    next_bids = len(re.findall(bid_pattern, next_state))
    
    bid_change = next_bids - state_bids
    
    # Bid growth indicates progress (more interactive elements available)
    bid_progress = 0.0
    if bid_change > 0:
        bid_progress = min(0.2, bid_change * 0.05)
    elif bid_change < 0:
        bid_progress = max(-0.1, bid_change * 0.02)  # Small penalty for losing elements
    
    # Form field indicators - filled fields suggest progress
    filled_count = len(re.findall(r"fill\s*\(\s*['\"]bid[^)]+['\"]\s*,\s*['\"]", action))
    form_progress = filled_count * 0.15
    
    # Calculate base Q-value from components
    base_q = completion_score + action_bonus + bid_progress + form_progress
    
    # Step efficiency consideration - fewer remaining steps means higher urgency
    # Estimate current step from state complexity (heuristic)
    # More complex state = later in episode
    state_complexity = len(state) // 500  # Rough estimate
    remaining_steps = max(1, 45 - min(state_complexity, 40))
    urgency_factor = remaining_steps / 45.0  # Higher when more steps remain
    
    # Adjust Q-value based on urgency (earlier progress is more valuable)
    adjusted_q = base_q * (1.0 + urgency_factor * 0.3)
    
    # Penalize unproductive actions when not near completion
    if is_noop and completion_score < 0.3:
        adjusted_q -= 0.1
    
    # Penalize excessive scrolling without progress
    if is_scroll and bid_change <= 0 and completion_score < 0.2:
        adjusted_q -= 0.05
    
    # Ensure value is in valid range
    q_value = max(0.0, min(1.0, adjusted_q))
    
    return q_value