def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for terminal success state
    success_patterns = ['task completed', 'successfully', 'congratulations', 'done', 'finished', 
                       'reached the goal', 'you have successfully']
    for pattern in success_patterns:
        if pattern in next_state_lower:
            return 1.0
    
    # Check for terminal failure state
    failure_patterns = ['max steps', 'step limit', 'failed', 'timeout', 'episode ended', 
                       'terminated', 'maximum steps']
    for pattern in failure_patterns:
        if pattern in next_state_lower:
            return 0.0
    
    # Estimate success probability based on state features
    success_prob = 0.0
    
    # Feature 1: Holding the target object (critical for most tasks)
    if 'holding' in state_lower or 'you are holding' in state_lower:
        success_prob += 0.3
    elif 'take' in action_lower or 'pickup' in action_lower:
        success_prob += 0.15
    
    # Feature 2: At or near target location
    target_locations = ['sinkbasin', 'countertop', 'diningtable', 'coffeetable', 'sidetable', 
                       'dresser', 'microwave', 'fridge', 'cabinet', 'drawer', 'shelf', 
                       'armchair', 'sofa', 'bed', 'toilet', 'bathtub', 'stoveburner', 
                       'garbagecan', 'sink', 'stove', 'oven', 'fridge']
    for loc in target_locations:
        if loc in next_state_lower:
            success_prob += 0.25
            break
    
    # Feature 3: Action is productive (not just moving around)
    productive_verbs = ['go', 'take', 'put', 'clean', 'heat', 'cool', 'open', 'close', 
                       'toggle', 'wash', 'putto', 'put in', 'put on']
    for verb in productive_verbs:
        if verb in action_lower:
            success_prob += 0.15
            break
    
    # Feature 4: State shows progression (more context than before)
    state_len = len(state_lower.split())
    next_len = len(next_state_lower.split())
    if next_len > state_len:
        success_prob += 0.1
    
    # Feature 5: Check for negative indicators (errors, nothing found)
    negative_patterns = ['nothing', 'cannot', 'cannot find', 'invalid', 'error', 
                        'no', 'not here', 'not found', 'does not exist']
    for pattern in negative_patterns:
        if pattern in next_state_lower:
            success_prob -= 0.2
            break
    
    # Feature 6: Room navigation progress (moving through rooms)
    rooms = ['kitchen', 'livingroom', 'bedroom', 'bathroom', 'office', 'garage', 'hallway']
    room_count = sum(1 for room in rooms if room in next_state_lower)
    if room_count > 0:
        success_prob += 0.05 * room_count
    
    # Normalize probability to [0, 1]
    success_prob = min(1.0, max(0.0, success_prob))
    
    # Q-value = expected return (success_prob * 1.0 reward for success)
    # Add small base value for non-terminal states to encourage exploration
    q_value = success_prob * 0.9 + 0.05
    
    return q_value