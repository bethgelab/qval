def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Base value for a neutral state
    value = 0.0
    
    # Check for task completion - immediate success
    completion_indicators = ['success', 'task completed', 'completed', 'done', 'finished']
    for indicator in completion_indicators:
        if indicator in state_lower:
            return 1.0
    
    # Check if agent is holding an object (progress indicator)
    holding_indicators = ['holding', 'take', 'pickup', 'pick up', 'you are carrying']
    for indicator in holding_indicators:
        if indicator in state_lower:
            value += 0.35
            break
    
    # Check if agent is at a receptacle/location (important for task completion)
    location_keywords = ['kitchen', 'bedroom', 'bathroom', 'living room', 'office',
                        'counter', 'table', 'shelf', 'drawer', 'fridge', 'microwave',
                        'sink', 'cabinet', 'closet', 'dresser', 'nightstand', 'chair',
                        'sofa', 'bed', 'desk', 'stove', 'oven', 'toaster', 'garbage',
                        'cupboard', 'pantry', 'microwave', 'cooler', 'sinkbasin']
    for keyword in location_keywords:
        if keyword in state_lower:
            value += 0.15
            break
    
    # Check for action verbs that indicate progress toward goal
    action_indicators = ['put', 'place', 'arrange', 'clean', 'heat', 'cool', 'slice']
    for indicator in action_indicators:
        if indicator in state_lower:
            value += 0.2
            break
    
    # Check if agent needs to search (indicates incomplete progress)
    search_indicators = ['nothing', 'cannot find', 'does not exist', 'not visible', 'empty']
    for indicator in search_indicators:
        if indicator in state_lower:
            value -= 0.15
            break
    
    # Check for navigation (necessary but not sufficient for success)
    nav_indicators = ['go', 'walk', 'move', 'navigate', 'enter']
    for indicator in nav_indicators:
        if indicator in state_lower:
            value += 0.1
            break
    
    # Penalize being in a potentially stuck state
    stuck_indicators = ['cannot', 'impossible', 'invalid', 'error', 'not available']
    for indicator in stuck_indicators:
        if indicator in state_lower:
            value -= 0.2
            break
    
    # Normalize value to [0, 1] range
    value = max(0.0, min(1.0, value))
    
    return value