def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check if goal is already achieved
    if any(phrase in state_lower for phrase in ['done', 'success', 'completed', 'finished', 'goal achieved', 'task complete']):
        return 1.0
    
    # Check for failure indicators
    if any(phrase in state_lower for phrase in ['failed', 'timeout', 'cannot', 'impossible', 'error']):
        return 0.0
    
    # Check if we're at the goal location
    if any(phrase in state_lower for phrase in ['at the', 'in the', 'standing at', 'location']):
        base_score = 0.5
    else:
        base_score = 0.3
    
    # Count positive progress indicators
    positive_patterns = [
        r'\bin\s+\w+\b',  # "in [object]"
        r'\bon\s+\w+\b',  # "on [object]"
        r'\bat\s+\w+\b',  # "at [location]"
        r'clean',
        r'open',
        r'on\b',
        r'filled',
        r'hot',
        r'charged',
        r'placed',
        r'moved'
    ]
    
    # Count negative indicators
    negative_patterns = [
        r'dirty',
        r'closed',
        r'off\b',
        r'empty',
        r'cold',
        r'broken',
        r'blocked',
        r'locked'
    ]
    
    positive_count = sum(1 for pattern in positive_patterns if re.search(pattern, state_lower))
    negative_count = sum(1 for pattern in negative_patterns if re.search(pattern, state_lower))
    
    # Adjust score based on object states
    state_adjustment = (positive_count - negative_count) * 0.05
    base_score += state_adjustment
    
    # Check for navigation progress
    if any(phrase in state_lower for phrase in ['walked', 'moved', 'navigate', 'go to']):
        base_score += 0.1
    
    # Check for object interaction progress
    if any(phrase in state_lower for phrase in ['took', 'picked', 'grabbed', 'held', 'carrying']):
        base_score += 0.15
    
    # Check if we have the target object
    if any(phrase in state_lower for phrase in ['holding', 'in inventory', 'have']):
        base_score += 0.1
    
    # Penalty for obstacles
    if any(phrase in state_lower for phrase in ['cannot', 'blocked', 'need to', 'must']):
        base_score -= 0.1
    
    # Clamp to valid range [0, 1]
    return max(0.0, min(1.0, base_score))