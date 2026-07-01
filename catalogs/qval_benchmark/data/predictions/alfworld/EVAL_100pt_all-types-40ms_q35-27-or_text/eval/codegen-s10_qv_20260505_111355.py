def signal_function(state: str, action: str, next_state: str) -> float:
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    
    q_value = 0.5  # Neutral baseline
    
    # Reward productive actions that typically progress toward goals
    productive_keywords = ['put', 'take', 'go to', 'turn on', 'turn off', 'clean', 'open', 'close']
    for keyword in productive_keywords:
        if keyword in action_lower:
            q_value += 0.1
            break
    
    # Reward progress indicators in the next state
    progress_keywords = ['on the', 'in the', 'at the', 'is clean', 'is open', 'is closed', 'is on', 'is off']
    for keyword in progress_keywords:
        if keyword in next_state_lower:
            q_value += 0.1
            break
    
    # Penalize failure indicators that suggest the action didn't work
    failure_keywords = ['cannot', 'nothing', 'empty', 'already', 'already is', 'already in']
    for keyword in failure_keywords:
        if keyword in next_state_lower:
            q_value -= 0.2
            break
    
    # Strongly reward goal completion indicators
    goal_keywords = ['done', 'success', 'completed', 'goal', 'finished']
    for keyword in goal_keywords:
        if keyword in next_state_lower:
            q_value += 0.4
            break
    
    # Bonus for successful object placement actions
    if 'put' in action_lower and ('on the' in next_state_lower or 'in the' in next_state_lower):
        q_value += 0.15
    
    # Small bonus for navigation actions that enable further progress
    if 'go to' in action_lower:
        q_value += 0.05
    
    # Check if objects are in target locations in the state
    target_patterns = ['on the countertop', 'on the table', 'in the drawer', 'in the fridge', 'in the microwave']
    for pattern in target_patterns:
        if pattern in state_lower:
            q_value += 0.05
            break
    
    # Clamp to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value