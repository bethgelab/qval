def signal_function(state: str) -> float:
    import re
    
    # Base value from sparse reward structure
    base_value = 0.0
    
    # Check for success indicators in state
    success_patterns = [
        r'task completed',
        r'task success',
        r'episode success',
        r'correct',
        r'completed successfully',
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 1.0
    
    # Check for goal-related objects mentioned as being in correct location
    goal_patterns = [
        r'placed',
        r'removed',
        r'cleaned',
        r'washed',
        r'sliced',
        r'cooled',
        r'heated',
        r'cooked',
    ]
    
    progress_count = 0
    for pattern in goal_patterns:
        matches = re.findall(pattern, state, re.IGNORECASE)
        progress_count += len(matches)
    
    # Check for object manipulation completion
    manipulation_patterns = [
        r'put',
        r'take',
        r'go to',
        r'look at',
        r'examine',
    ]
    
    action_count = 0
    for pattern in manipulation_patterns:
        matches = re.findall(pattern, state, re.IGNORECASE)
        action_count += len(matches)
    
    # Check for remaining steps budget (estimate from step limit context)
    step_limit = 40
    estimated_remaining = step_limit
    
    # If state mentions steps taken or remaining
    step_patterns = [
        r'step\s+(\d+)',
        r'(\d+)\s*steps?\s*remaining',
        r'(\d+)\s*steps?\s*left',
    ]
    
    for pattern in step_patterns:
        matches = re.findall(pattern, state, re.IGNORECASE)
        if matches:
            for match in matches:
                try:
                    num = int(match[0])
                    if 'remaining' in pattern or 'left' in pattern:
                        estimated_remaining = num
                    elif 'step' in pattern:
                        estimated_remaining = step_limit - num
                except (ValueError, IndexError):
                    pass
    
    # Calculate progress score based on goal-related actions completed
    progress_score = min(progress_count * 0.2, 0.6)
    
    # Calculate action efficiency score (fewer actions needed = better)
    efficiency_score = max(0.0, 1.0 - (action_count * 0.05))
    efficiency_score = min(efficiency_score, 1.0)
    
    # Calculate remaining steps score (more steps = better chance)
    steps_score = min(estimated_remaining / step_limit, 1.0)
    
    # Check for potential obstacles or errors
    error_patterns = [
        r'error',
        r'invalid',
        r'cannot',
        r'not found',
        r'empty',
        r'missing',
        r'blocked',
    ]
    
    obstacle_penalty = 0.0
    for pattern in error_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            obstacle_penalty += 0.15
    
    obstacle_penalty = min(obstacle_penalty, 0.5)
    
    # Combine scores with weighted factors
    combined_value = (
        progress_score * 0.4 +
        efficiency_score * 0.3 +
        steps_score * 0.3 -
        obstacle_penalty
    )
    
    # Ensure value is in valid range
    combined_value = max(0.0, min(1.0, combined_value))
    
    return combined_value