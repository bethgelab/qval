def signal_function(state: str) -> float:
    """
    Estimate the state-value for ALFWorld environment based on state analysis.
    
    Args:
        state: Text representation of the current state
        
    Returns:
        Estimated state-value between 0.0 and 1.0
    """
    import re
    
    state_lower = state.lower()
    
    # Check if goal is already achieved
    success_indicators = ['done', 'success', 'completed', 'task complete', 'goal achieved', 'goal accomplished']
    for indicator in success_indicators:
        if indicator in state_lower:
            return 0.95
    
    # Check for failure indicators
    failure_indicators = ['failed', 'cannot', 'impossible', 'error', 'out of steps']
    for indicator in failure_indicators:
        if indicator in state_lower:
            return 0.1
    
    # Parse for step information if available
    step_match = re.search(r'step[:\s]*(\d+)', state_lower)
    if step_match:
        current_step = int(step_match.group(1))
        steps_remaining = 40 - current_step
        if steps_remaining <= 0:
            return 0.05
    else:
        steps_remaining = 40
    
    # Estimate progress based on object states and task completion indicators
    progress_score = 0.0
    
    # Check for objects in correct locations or states
    location_indicators = ['on the', 'in the', 'at the', 'inside the', 'contains the']
    for indicator in location_indicators:
        if indicator in state_lower:
            progress_score += 0.08
    
    # Check for completed subtasks
    subtask_indicators = ['cleaned', 'opened', 'moved', 'placed', 'turned on', 'turned off', 'warmed', 'cooled']
    for indicator in subtask_indicators:
        if indicator in state_lower:
            progress_score += 0.12
    
    # Check for goal-related objects mentioned
    goal_objects = ['sink', 'microwave', 'fridge', 'stove', 'countertop', 'table', 'cupboard', 'drawer']
    for obj in goal_objects:
        if obj in state_lower:
            progress_score += 0.05
    
    # Cap progress score
    progress_score = min(progress_score, 0.7)
    
    # Estimate value based on progress and remaining steps
    step_factor = min(steps_remaining / 40.0, 1.0)
    
    # Combine factors with emphasis on remaining steps
    estimated_value = 0.4 * progress_score + 0.6 * step_factor
    
    # Add bonus for positive accessibility indicators
    positive_indicators = ['can', 'able', 'available', 'reachable', 'accessible', 'open', 'unlocked']
    for indicator in positive_indicators:
        if indicator in state_lower:
            estimated_value = min(estimated_value + 0.03, 0.95)
    
    # Penalty for negative indicators
    negative_indicators = ['closed', 'locked', 'dirty', 'broken', 'missing']
    for indicator in negative_indicators:
        if indicator in state_lower:
            estimated_value = max(estimated_value - 0.05, 0.0)
    
    # Ensure value is in valid range
    return max(0.0, min(1.0, estimated_value))