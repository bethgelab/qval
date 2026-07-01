def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check for task completion (immediate success)
    success_patterns = [
        'task completed',
        'successfully',
        'congratulations',
        'task is complete',
        'you have completed',
        'all done'
    ]
    
    for pattern in success_patterns:
        if pattern.lower() in next_state.lower():
            return 1.0
    
    # Check for invalid/unproductive actions
    error_patterns = [
        'you cannot',
        'nothing happens',
        'invalid',
        'error',
        'cannot see',
        'cannot find',
        'cannot take',
        'cannot put',
        'not in',
        'not here'
    ]
    
    for pattern in error_patterns:
        if pattern.lower() in next_state.lower():
            return -0.3
    
    # Analyze progress indicators in states
    progress_score = 0.0
    
    # Check for object acquisition (holding items)
    holding_patterns = [
        'you are holding',
        'you have in your hands',
        'holding'
    ]
    
    state_holding = any(p in state.lower() for p in holding_patterns)
    next_holding = any(p in next_state.lower() for p in holding_patterns)
    
    if next_holding and not state_holding:
        progress_score += 0.4
    
    # Check for object placement (on surfaces/containers)
    state_placement = len(re.findall(r'\w+\s+on\s+(?:the\s+)?\w+', state.lower()))
    next_placement = len(re.findall(r'\w+\s+on\s+(?:the\s+)?\w+', next_state.lower()))
    
    if next_placement > state_placement:
        progress_score += 0.3
    
    # Check for container interactions (open/close)
    container_patterns = ['open', 'close', 'drawer', 'cupboard', 'fridge', 'microwave']
    state_containers = sum(1 for p in container_patterns if p in state.lower())
    next_containers = sum(1 for p in container_patterns if p in next_state.lower())
    
    if next_containers > state_containers:
        progress_score += 0.2
    
    # Check for location changes (room transitions)
    room_keywords = ['bedroom', 'kitchen', 'living room', 'bathroom', 'office', 'garage', 'basement']
    state_rooms = sum(1 for room in room_keywords if room in state.lower())
    next_rooms = sum(1 for room in room_keywords if room in next_state.lower())
    
    if next_rooms > state_rooms:
        progress_score += 0.1
    
    # Action type analysis
    action_lower = action.lower()
    useful_actions = ['take', 'put', 'open', 'close', 'clean', 'heat', 'cool', 'charge', 'toggle']
    navigation_actions = ['go to', 'walk to', 'move to', 'enter']
    
    if any(a in action_lower for a in useful_actions):
        progress_score += 0.2
    
    if any(a in action_lower for a in navigation_actions):
        progress_score += 0.1
    
    # Penalize actions that don't make progress
    if progress_score == 0.0:
        progress_score = -0.1
    
    # Calculate Q-value with appropriate bounds
    # Base value reflects potential for future success
    q_value = 0.1 + progress_score
    
    # Bound the value appropriately for sparse reward setting
    q_value = max(-0.5, min(0.8, q_value))
    
    return q_value