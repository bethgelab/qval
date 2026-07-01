def signal_function(state: str) -> float:
    """
    Estimate the state-value V(s) for an ALFWorld environment state.
    
    The value represents the estimated probability of reaching the goal
    from the current state, assuming optimal play. Returns a float in [0, 1].
    """
    import re
    import math
    
    state_lower = state.lower()
    
    # Extract step count if available
    step_match = re.search(r'(\d+)\s*step', state_lower)
    steps_remaining = 40
    if step_match:
        steps_taken = int(step_match.group(1))
        steps_remaining = max(0, 40 - steps_taken)
    
    # Extract goal information
    goal_match = re.search(r'goal[:\s]+(.+?)(?:\s+goal|$)', state_lower)
    goal_text = goal_match.group(1).strip() if goal_match else ""
    
    # Check for success indicators
    success_keywords = ['done', 'completed', 'success', 'successfully', 'task completed']
    has_success = any(kw in state_lower for kw in success_keywords)
    
    # Check for failure indicators
    failure_keywords = ['failed', 'error', 'cannot', 'unable', 'blocked']
    has_failure = any(kw in state_lower for kw in failure_keywords)
    
    # Check if agent is at a relevant location
    location_match = re.search(r'you are at\s+(\w+)', state_lower)
    current_location = location_match.group(1) if location_match else "unknown"
    
    # Check for object mentions (key items)
    object_mentions = re.findall(r'\b(the|a|an)\s+(\w+)', state_lower)
    
    # Base value estimation
    base_value = 0.5
    
    # Adjust for success/failure indicators
    if has_success:
        base_value = 0.95
    elif has_failure:
        base_value = 0.1
    
    # Adjust for steps remaining (more steps = higher potential)
    step_factor = steps_remaining / 40.0
    base_value = base_value * (0.3 + 0.7 * step_factor)
    
    # Adjust for goal clarity
    if goal_text:
        base_value = min(1.0, base_value + 0.1)
    
    # Adjust for location information
    if location_match and current_location not in ['unknown', '']:
        base_value = min(1.0, base_value + 0.05)
    
    # Adjust for object mentions (more objects = more complex task)
    if len(object_mentions) > 0:
        base_value = min(1.0, base_value - 0.05)
    
    # Clamp value to [0, 1]
    base_value = max(0.0, min(1.0, base_value))
    
    # Add small noise based on state complexity
    complexity = len(state) / 200.0
    noise = 0.02 * (1 - complexity)
    
    return round(base_value + noise, 4)