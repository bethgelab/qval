def signal_function(state: str) -> float:
    """
    Estimates the state-value for an ALFWorld state.
    
    Args:
        state: Text representation of the current state
        
    Returns:
        Estimated state-value (float between 0 and 1)
    """
    import re
    
    state_lower = state.lower()
    
    # Check for task completion indicators
    completion_indicators = [
        'task completed', 'success', 'done', 'finished',
        'you have completed', 'goal achieved', 'task is completed',
        'successfully completed', 'episode successful'
    ]
    for indicator in completion_indicators:
        if indicator in state_lower:
            return 1.0
    
    # Check for failure indicators
    failure_indicators = [
        'failed', 'timeout', 'step limit', 'exceeded', 'cannot'
    ]
    for indicator in failure_indicators:
        if indicator in state_lower:
            return 0.0
    
    # Check for step count to estimate remaining steps
    step_count = 0
    step_match = re.search(r'(?:step|steps?)[\s:]+(\d+)', state_lower)
    if step_match:
        try:
            step_count = int(step_match.group(1))
        except:
            pass
    
    # Estimate remaining steps
    step_limit = 40
    remaining_steps = max(0, step_limit - step_count)
    step_factor = remaining_steps / step_limit
    
    # Check for possession of objects
    possession_score = 0
    possession_patterns = [
        r'you have the (\w+)',
        r'you are holding the (\w+)',
        r'you have (\w+)',
        r'holding the (\w+)',
        r'in your hand (\w+)'
    ]
    for pattern in possession_patterns:
        matches = re.findall(pattern, state_lower)
        if matches:
            possession_score += len(matches) * 0.15
    
    # Check for object in target location (moved to correct place)
    location_score = 0
    location_patterns = [
        r'(\w+) is on the (\w+)',
        r'(\w+) is in the (\w+)',
        r'(\w+) is at the (\w+)',
        r'(\w+) on (\w+)',
        r'(\w+) in (\w+)'
    ]
    for pattern in location_patterns:
        matches = re.findall(pattern, state_lower)
        if matches:
            location_score += len(matches) * 0.1
    
    # Check for object state changes (cleaned, opened, etc.)
    state_score = 0
    state_patterns = [
        r'(\w+) is clean',
        r'(\w+) is open',
        r'(\w+) is closed',
        r'(\w+) is turned on',
        r'(\w+) is turned off',
        r'(\w+) is filled',
        r'(\w+) is empty'
    ]
    for pattern in state_patterns:
        matches = re.findall(pattern, state_lower)
        if matches:
            state_score += len(matches) * 0.12
    
    # Check for being in relevant location
    location_present = 0
    location_keywords = ['kitchen', 'bedroom', 'bathroom', 'livingroom', 'office', 
                         'diningroom', 'garden', 'closet', 'shelf', 'counter', 
                         'table', 'cabinet', 'drawer', 'sink', 'stove', 'fridge',
                         'microwave', 'oven', 'toaster']
    for keyword in location_keywords:
        if keyword in state_lower:
            location_present += 0.03
    
    # Check for goal-related objects mentioned in state
    goal_object_score = 0
    goal_keywords = ['toaster', 'microwave', 'oven', 'fridge', 'sink', 'soap', 
                     'bottle', 'cup', 'bowl', 'plate', 'spoon', 'fork', 'knife', 
                     'cloth', 'cleaner', 'spray', 'bucket', 'mop', 'towel',
                     'batteries', 'battery', 'light', 'switch', 'key', 'box',
                     'container', 'jar', 'can', 'bag', 'paper', 'pen', 'book']
    for keyword in goal_keywords:
        if keyword in state_lower:
            goal_object_score += 0.02
    
    # Check for navigation progress (moved between rooms)
    nav_score = 0
    nav_keywords = ['go to', 'move to', 'walk to', 'navigate to', 'go through',
                    'enter the', 'exit the', 'opened the door', 'closed the door']
    for keyword in nav_keywords:
        if keyword in state_lower:
            nav_score += 0.05
    
    # Check for interaction success
    interaction_score = 0
    interaction_patterns = [
        r'you (?:picked|put|took|moved|cleaned|opened|closed|turned) (?:the|a)',
        r'(?:picked|put|took|moved|cleaned|opened|closed|turned) up',
        r'you (?:picked|put|took|moved|cleaned|opened|closed|turned) (\w+)'
    ]
    for pattern in interaction_patterns:
        matches = re.findall(pattern, state_lower)
        if matches:
            interaction_score += len(matches) * 0.08
    
    # Combine scores with weights
    base_value = 0.15
    
    value = (
        base_value +
        possession_score * 0.35 +
        location_score * 0.25 +
        state_score * 0.20 +
        location_present * 0.05 +
        goal_object_score * 0.05 +
        nav_score * 0.05 +
        interaction_score * 0.05
    )
    
    # Adjust by remaining steps (efficiency matters)
    value = value * (0.5 + 0.5 * step_factor)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, value))