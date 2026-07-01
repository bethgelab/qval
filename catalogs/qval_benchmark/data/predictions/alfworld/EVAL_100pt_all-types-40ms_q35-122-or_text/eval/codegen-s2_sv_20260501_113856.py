def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for terminal success
    success_terms = ['success', 'completed', 'finished', 'task done', 'congratulations', 'you have completed']
    if any(term in state_lower for term in success_terms):
        return 1.0
    
    # Check for terminal failure
    failure_terms = ['failed', 'timeout', 'exceeded', 'error', 'cannot complete']
    if any(term in state_lower for term in failure_terms):
        return 0.0
    
    # Start with base value for non-terminal states
    value = 0.15
    
    # Check if agent is carrying something (key progress indicator)
    carrying_patterns = [
        r'carrying.*?:\s*(.+?)(?:\n|$)',
        r'holding.*?:\s*(.+?)(?:\n|$)',
        r'inventory.*?:\s*(.+?)(?:\n|$)'
    ]
    for pattern in carrying_patterns:
        match = re.search(pattern, state_lower)
        if match:
            items = match.group(1).strip()
            if items and items != 'nothing':
                value += 0.4
                break
    
    # Check for room location (being in a room is progress)
    room_keywords = ['kitchen', 'bedroom', 'bathroom', 'livingroom', 'garage', 'basement', 'office', 'diningroom', 'hallway']
    rooms_found = sum(1 for room in room_keywords if room in state_lower)
    if rooms_found > 0:
        value += min(0.25, rooms_found * 0.05)
    
    # Check for object mentions (exploration progress)
    object_patterns = ['on the', 'in the', 'under the', 'next to the', 'beside the', 'near the']
    objects_found = sum(1 for pattern in object_patterns if pattern in state_lower)
    value += min(0.15, objects_found * 0.02)
    
    # Penalize for negative feedback
    negative_patterns = ['cannot', 'nothing', 'empty', 'not found', 'no such', 'invalid']
    negatives_found = sum(1 for pattern in negative_patterns if pattern in state_lower)
    value -= min(0.2, negatives_found * 0.05)
    
    # Clamp to valid range
    return max(0.0, min(1.0, value))