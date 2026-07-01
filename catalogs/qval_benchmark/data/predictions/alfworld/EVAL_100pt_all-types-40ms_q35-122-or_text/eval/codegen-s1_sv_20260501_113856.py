def signal_function(state: str) -> float:
    """
    Estimate state-value for ALFWorld environment.
    Returns a value in [0, 1] representing expected discounted cumulative reward.
    """
    state_lower = state.lower()
    
    # Check for task completion - highest value
    completion_indicators = [
        "your task is completed",
        "task completed",
        "completed successfully",
        "success"
    ]
    
    for indicator in completion_indicators:
        if indicator in state_lower:
            return 1.0
    
    # Check for failure indicators
    failure_indicators = [
        "cannot do that",
        "cannot find",
        "nothing to",
        "nothing here",
        "invalid"
    ]
    
    failure_score = 0
    for indicator in failure_indicators:
        if indicator in state_lower:
            failure_score += 0.1
    
    # Check for progress indicators
    # Objects in inventory
    inventory_match = re.search(r'you are holding: (.+?)(?:\n|$)', state, re.IGNORECASE)
    has_object = False
    if inventory_match:
        holding = inventory_match.group(1).strip()
        if holding and holding.lower() != 'nothing':
            has_object = True
    
    # Check for task-relevant actions
    action_indicators = [
        "put", "take", "go", "open", "close", "clean", "heat", "cool", "find"
    ]
    
    action_count = sum(1 for action in action_indicators if action in state_lower)
    
    # Check for room exploration (indicates progress)
    room_keywords = ['kitchen', 'living room', 'bedroom', 'bathroom', 'garage', 'office', 'dining room']
    rooms_explored = sum(1 for room in room_keywords if room in state_lower)
    
    # Base value calculation
    base_value = 0.15  # Starting value for any valid state
    
    # Add value for having objects (progress toward goal)
    if has_object:
        base_value += 0.25
    
    # Add value for taking actions (engagement with environment)
    base_value += min(action_count * 0.05, 0.25)
    
    # Add value for room exploration
    if rooms_explored > 1:
        base_value += 0.1
    elif rooms_explored > 0:
        base_value += 0.05
    
    # Penalize for failures
    base_value -= failure_score
    
    # Ensure value is in [0, 1]
    return max(0.0, min(1.0, base_value))