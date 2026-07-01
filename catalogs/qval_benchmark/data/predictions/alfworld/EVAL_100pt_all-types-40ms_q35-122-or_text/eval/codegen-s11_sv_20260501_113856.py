def signal_function(state: str) -> float:
    import re
    
    # Check for task completion (immediate success)
    success_patterns = [
        r'task completed',
        r'successfully',
        r'task finished',
        r'congratulations',
        r'you win',
        r'thank you for playing',
        r'task done',
        r'episode finished'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 1.0
    
    # Check for failure conditions
    failure_patterns = [
        r'task failed',
        r'you lose',
        r'game over',
        r'too many steps',
        r'step limit reached'
    ]
    
    for pattern in failure_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 0.0
    
    # Extract step count if available
    step_match = re.search(r'step (\d+)', state)
    current_step = 0
    if step_match:
        current_step = int(step_match.group(1))
    remaining_steps = 40 - current_step
    
    # Base value from remaining steps (more steps = more opportunity)
    base_value = remaining_steps / 40.0
    
    # Progress indicators in state text
    score = 0.0
    
    # Object pickup progress (high value - key action)
    pickup_patterns = [
        r'you pick up',
        r'you take',
        r'you grab',
        r'you have',
        r'holding',
        r'inventory contains'
    ]
    for pattern in pickup_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            score += 0.25
            break
    
    # Object placement progress (high value - near completion)
    placement_patterns = [
        r'you put',
        r'you place',
        r'you drop',
        r'you have put'
    ]
    for pattern in placement_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            score += 0.35
            break
    
    # Room navigation progress (medium value)
    room_patterns = [
        r'you arrive at',
        r'you enter',
        r'you go to',
        r'you are in'
    ]
    for pattern in room_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            score += 0.15
            break
    
    # Cleaning/heating/cooling progress (task-specific)
    state_change_patterns = [
        r'you clean',
        r'you heat',
        r'you cool',
        r'you wipe',
        r'you turn on',
        r'you turn off'
    ]
    for pattern in state_change_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            score += 0.2
            break
    
    # Combine base value with progress bonus
    # Weight progress more heavily when steps are available
    progress_weight = min(1.0, remaining_steps / 20.0)
    estimated_value = base_value * 0.3 + score * progress_weight
    
    # Penalize if very few steps remain without clear progress
    if remaining_steps < 5 and score < 0.5:
        estimated_value *= 0.5
    
    # Clamp to valid range
    return max(0.0, min(1.0, estimated_value))