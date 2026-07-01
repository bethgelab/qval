def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    # Extract bid numbers from state strings
    def extract_bids(text):
        return set(re.findall(r'\d+', text))
    
    state_bids = extract_bids(state)
    next_bids = extract_bids(next_state)
    
    # Check if action appears valid based on element references
    action_valid = False
    if action.startswith('click('):
        element_ref = action[6:-1].strip()
        if element_ref in state or element_ref in next_state:
            action_valid = True
    elif action.startswith('fill('):
        element_ref = action[5:-1].strip()
        if element_ref in state or element_ref in next_state:
            action_valid = True
    elif action.startswith('press('):
        action_valid = True
    elif action.startswith('noop'):
        action_valid = True
    elif action.startswith('scroll('):
        action_valid = True
    
    # If action is invalid, penalize heavily
    if not action_valid:
        return -0.8
    
    # Check for state progression
    state_changed = state != next_state
    
    # Check for increased element visibility (more bids in next state)
    if len(next_bids) > len(state_bids):
        bid_progress = 0.3
    elif len(next_bids) == len(state_bids):
        bid_progress = 0.1
    else:
        bid_progress = 0.0
    
    # Check for goal-related indicators in next state
    goal_indicators = ['success', 'complete', 'done', 'achieved', 'saved', 'created', 'sent']
    goal_keywords_present = sum(1 for kw in goal_indicators if kw in next_state.lower())
    goal_bonus = min(0.4, goal_keywords_present * 0.1)
    
    # Check for error indicators in next state
    error_indicators = ['error', 'failed', 'invalid', 'missing', 'not found', 'cannot']
    error_keywords_present = sum(1 for kw in error_indicators if kw in next_state.lower())
    error_penalty = min(0.5, error_keywords_present * 0.2)
    
    # Check for form interaction progress
    form_progress = 0.0
    if 'filled' in next_state.lower() or 'selected' in next_state.lower() or 'checked' in next_state.lower():
        form_progress = 0.2
    
    # Check for navigation progress
    nav_progress = 0.0
    if state_changed and len(next_bids) > len(state_bids):
        nav_progress = 0.2
    
    # Compute base Q-value
    base_value = bid_progress + goal_bonus - error_penalty + form_progress + nav_progress
    
    # Apply state change bonus (avoiding loops)
    if state_changed:
        base_value += 0.1
    
    # Apply action validity bonus
    if action_valid:
        base_value += 0.1
    
    # Clamp to reasonable range for Q-value estimation
    return max(-0.5, min(1.0, base_value))