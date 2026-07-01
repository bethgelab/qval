def signal_function(state: str) -> float:
    """
    Estimate state-value for ALFWorld environment based on text state representation.
    Returns a float between 0.0 and 1.0 representing estimated expected return.
    """
    import re
    
    state_lower = state.lower()
    
    # Check for terminal success state
    success_patterns = [
        r'you have done it',
        r'task completed',
        r'success',
        r'congratulations',
        r'well done',
        r'task is complete',
        r'you have completed the task'
    ]
    for pattern in success_patterns:
        if re.search(pattern, state_lower):
            return 1.0
    
    # Check for failure/terminal states
    failure_patterns = [
        r'step limit',
        r'out of steps',
        r'failed',
        r'cannot find',
        r'not found'
    ]
    for pattern in failure_patterns:
        if re.search(pattern, state_lower):
            return 0.0
    
    # Check if agent is holding an object
    holding = bool(re.search(r'holding\s+\w+', state_lower)) or \
              bool(re.search(r'carrying\s+\w+', state_lower))
    
    # Check current location
    room_keywords = ['kitchen', 'bedroom', 'livingroom', 'garage', 'bathroom', 'office', 
                     'diningroom', 'sidetable', 'drawer', 'cabinet', 'countertop']
    current_room = None
    for room in room_keywords:
        if room in state_lower:
            current_room = room
            break
    
    # Check if target object is visible in state
    target_found = bool(re.search(r'you see', state_lower)) and \
                   bool(re.search(r'tomato|apple|bowl|plate|cup|mug|pen|pencil|book|key|cloth', state_lower))
    
    # Check if agent has navigated (progress indicator)
    navigated = bool(re.search(r'you go to|you arrive at', state_lower))
    
    # Check for empty locations (negative indicator)
    empty_locations = bool(re.search(r'nothing|empty', state_lower))
    
    # Estimate value based on progress indicators
    value = 0.2  # Base value for non-terminal states
    
    # Significant value if holding an object
    if holding:
        value = 0.7
    
    # Increase if in a known room
    if current_room:
        value += 0.1
    
    # Increase if target object found
    if target_found:
        value += 0.15
    
    # Increase if agent has navigated
    if navigated:
        value += 0.05
    
    # Decrease if encountering empty locations
    if empty_locations:
        value -= 0.05
    
    # Adjust based on whether we're stuck
    if 'nothing' in state_lower and not holding:
        value -= 0.05
    
    # Ensure value stays in valid range
    value = max(0.0, min(1.0, value))
    
    return value