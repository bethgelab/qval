def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for task completion indicators
    completion_keywords = ['done', 'success', 'completed', 'task complete', 'goal achieved']
    for keyword in completion_keywords:
        if keyword in state_lower:
            return 1.0
    
    # Check for failure indicators
    failure_keywords = ['fail', 'impossible', 'cannot', 'out of bounds', 'error']
    for keyword in failure_keywords:
        if keyword in state_lower:
            return 0.0
    
    # Start with a neutral base value
    value = 0.5
    
    # Count positive progress indicators
    positive_indicators = [
        'has ', 'holding ', 'carrying ', 'found ', 'picked up ',
        'opened ', 'closed ', 'turned on ', 'turned off ', 'cleaned ',
        'heated ', 'cooled ', 'placed ', 'put ', 'moved '
    ]
    positive_count = sum(1 for ind in positive_indicators if ind in state_lower)
    value += positive_count * 0.05
    
    # Check if key objects are accessible
    accessibility_keywords = ['can ', 'is ', 'available', 'near', 'close']
    accessibility_count = sum(1 for kw in accessibility_keywords if kw in state_lower)
    value += accessibility_count * 0.03
    
    # Check for blocking conditions
    blocking_keywords = ['blocked', 'locked', 'cannot', 'not accessible', 'missing']
    blocking_count = sum(1 for kw in blocking_keywords if kw in state_lower)
    value -= blocking_count * 0.05
    
    # Check for navigation progress
    nav_keywords = ['go to', 'navigate', 'walk', 'move to', 'approach']
    nav_count = sum(1 for kw in nav_keywords if kw in state_lower)
    value += nav_count * 0.02
    
    # Check if agent is in a relevant location
    location_keywords = ['in the', 'at the', 'on the', 'near the']
    location_count = sum(1 for kw in location_keywords if kw in state_lower)
    value += location_count * 0.02
    
    # Penalize if state seems stuck or repetitive
    stuck_keywords = ['nothing', 'empty', 'no', 'none']
    stuck_count = sum(1 for kw in stuck_keywords if kw in state_lower)
    value -= stuck_count * 0.02
    
    # Ensure value stays within valid range
    value = max(0.0, min(1.0, value))
    
    return value