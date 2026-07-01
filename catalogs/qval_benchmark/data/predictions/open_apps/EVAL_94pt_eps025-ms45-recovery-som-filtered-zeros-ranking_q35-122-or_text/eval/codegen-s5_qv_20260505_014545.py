def signal_function(state: str, action: str, next_state: str) -> float:
    """Estimate Q-value for OpenApps environment based on state analysis."""
    
    # Initialize base Q-value
    q_value = 0.0
    
    # Check for task completion indicators in next_state
    completion_patterns = [
        'success', 'completed', 'saved', 'sent', 'added', 'created',
        'confirmation', 'done', 'submitted', 'task complete', 'event created',
        'message sent', 'todo added', 'calendar event', 'code saved'
    ]
    
    next_state_lower = next_state.lower()
    for pattern in completion_patterns:
        if pattern in next_state_lower:
            q_value = 1.0
            return q_value
    
    # Evaluate action quality
    action_lower = action.lower()
    
    # Productive actions indicate progress
    productive_actions = ['fill', 'click', 'submit', 'press', 'enter', 'save', 'send']
    if any(act in action_lower for act in productive_actions):
        q_value += 0.15
    
    # Noop actions provide no progress signal
    if 'noop' in action_lower:
        q_value -= 0.05
    
    # Scrolling can be productive for finding elements
    if 'scroll' in action_lower:
        q_value += 0.05
    
    # Analyze state content for progress indicators
    state_content = (state + next_state).lower()
    
    # Count interactive elements (bids indicate clickable/fillable items)
    import re
    bid_count = len(re.findall(r"bid[0-9]+", state_content))
    
    # More bids = more actionable elements = better progress potential
    q_value += min(bid_count * 0.01, 0.2)
    
    # Check for form/input presence (indicates task progress)
    form_indicators = ['input', 'form', 'field', 'text', 'textarea', 'select']
    form_score = sum(1 for ind in form_indicators if ind in state_content)
    q_value += form_score * 0.03
    
    # Check for navigation progress (page transitions)
    page_indicators = ['page', 'tab', 'view', 'screen', 'window']
    page_score = sum(1 for ind in page_indicators if ind in state_content)
    q_value += page_score * 0.02
    
    # Penalize if state content shrinks significantly (possible error state)
    if len(next_state) < len(state) * 0.7:
        q_value -= 0.1
    
    # Penalize if action doesn't match available elements
    # Extract bid from action if present
    action_bid = re.search(r"bid[0-9]+", action_lower)
    if action_bid:
        bid_id = action_bid.group()
        if bid_id not in state_content:
            q_value -= 0.15  # Action targets non-existent element
    
    # Normalize Q-value to [0, 1] range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value