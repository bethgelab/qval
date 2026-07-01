def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check if goal appears to be achieved in next_state
    goal_indicators = [
        'success', 'completed', 'added', 'created', 'sent', 'updated', 
        'saved', 'done', 'finished', 'confirmed', 'scheduled', 'posted',
        'event added', 'message sent', 'task created', 'appointment created'
    ]
    next_lower = next_state.lower()
    if any(indicator in next_lower for indicator in goal_indicators):
        return 0.95
    
    # Check for errors or failures in next_state
    error_indicators = [
        'error', 'failed', 'invalid', 'denied', 'rejected', 'timeout',
        'not found', 'cannot', 'unable', 'missing', 'required'
    ]
    if any(indicator in next_lower for indicator in error_indicators):
        return 0.1
    
    # Evaluate action effectiveness
    action_lower = action.lower()
    
    # Productive actions (direct interaction)
    if 'fill' in action_lower:
        action_score = 0.55
    elif 'click' in action_lower:
        action_score = 0.5
    elif 'press' in action_lower:
        action_score = 0.45
    elif 'scroll' in action_lower:
        action_score = 0.25
    elif 'noop' in action_lower:
        action_score = 0.15
    else:
        action_score = 0.35
    
    # Detect state progression (meaningful changes between state and next_state)
    state_lower = state.lower()
    
    # Check if new interactive elements appeared
    bid_pattern = r"bid=\d+"
    state_bids = set(re.findall(bid_pattern, state))
    next_bids = set(re.findall(bid_pattern, next_state))
    
    if len(next_bids) > len(state_bids):
        action_score += 0.1  # New elements appeared (progress)
    elif len(next_bids) < len(state_bids) * 0.7:
        action_score -= 0.15  # Many elements disappeared (possibly wrong path)
    
    # Look for task-specific progress keywords
    progress_keywords = [
        'event', 'message', 'todo', 'task', 'item', 'entry', 'appointment',
        'meeting', 'calendar', 'contact', 'recipient', 'subject', 'date',
        'time', 'description', 'reminder', 'location'
    ]
    progress_count = sum(1 for kw in progress_keywords if kw in next_lower)
    action_score += min(progress_count * 0.03, 0.2)
    
    # Check for form completion indicators
    form_indicators = ['filled', 'entered', 'selected', 'chosen', 'picked']
    if any(ind in next_lower for ind in form_indicators):
        action_score += 0.08
    
    # Penalize if state didn't change much (stuck in same place)
    if next_state == state:
        action_score -= 0.3
    
    # Cap the value between 0 and 1
    return max(0.0, min(1.0, action_score))