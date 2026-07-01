def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for goal completion
    if 'success' in state_lower or 'completed' in state_lower or 'task completed' in state_lower:
        return 1.0
    
    # Check for failure indicators
    if 'failed' in state_lower or 'timeout' in state_lower or 'episode ended' in state_lower:
        return 0.0
    
    # Estimate remaining steps ratio
    steps_match = re.search(r'step[:\s]+(\d+)', state_lower)
    if steps_match:
        steps_taken = int(steps_match.group(1))
        remaining_ratio = max(0.0, 1.0 - steps_taken / 40.0)
    else:
        remaining_ratio = 1.0
    
    # Base value starts at 0.2 (some potential in any non-terminal state)
    value = 0.2
    
    # Check for positive progress indicators
    # Object manipulation progress
    if 'picked' in state_lower or 'taken' in state_lower:
        value += 0.15
    if 'holding' in state_lower:
        value += 0.1
    
    # Navigation progress
    if 'went to' in state_lower or 'moved to' in state_lower:
        value += 0.1
    
    # Object placement progress
    if 'placed' in state_lower or 'put' in state_lower:
        value += 0.15
    if 'cleaned' in state_lower:
        value += 0.1
    
    # Goal proximity indicators
    if 'target' in state_lower or 'destination' in state_lower:
        value += 0.1
    if 'goal' in state_lower:
        value += 0.05
    
    # Check if agent is in a relevant room
    rooms = ['kitchen', 'bedroom', 'bathroom', 'livingroom', 'office']
    for room in rooms:
        if room in state_lower:
            value += 0.05
            break
    
    # Check for object-goal alignment (objects in target locations)
    if 'on' in state_lower and ('counter' in state_lower or 'table' in state_lower or 'shelf' in state_lower):
        value += 0.1
    
    # Apply remaining steps multiplier (efficiency matters)
    value = value * (0.5 + 0.5 * remaining_ratio)
    
    # Cap value between 0 and 1
    return max(0.0, min(1.0, value))