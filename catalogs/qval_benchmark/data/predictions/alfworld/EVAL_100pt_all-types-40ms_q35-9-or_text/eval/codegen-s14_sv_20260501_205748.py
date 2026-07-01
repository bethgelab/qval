def signal_function(state: str) -> float:
    import re
    
    # Base value for incomplete task
    base_value = 0.0
    
    # Check if task appears complete (goal achieved)
    # Look for success indicators in state text
    success_patterns = [
        r'success',
        r'task completed',
        r'task complete',
        r'goal achieved',
        r'task done',
        r'object placed',
        r'placed in',
        r'cleaned',
        r'clean',
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 1.0
    
    # Check for failure or blocked indicators
    failure_patterns = [
        r'failed',
        r'cannot',
        r'blocked',
        r'error',
        r'limit reached',
        r'time out',
    ]
    
    for pattern in failure_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 0.0
    
    # Extract step information if present
    step_match = re.search(r'(\d+)', state)
    if step_match:
        current_step = int(step_match.group(1))
        step_limit = 40
        # Penalize being far from step limit
        steps_remaining = step_limit - current_step
        # Higher value if more steps remain (assuming we're making progress)
        step_score = max(0, steps_remaining / step_limit)
    else:
        step_score = 0.5
    
    # Check for goal-related keywords indicating progress
    goal_keywords = [
        'goal', 'target', 'destination', 'place', 'put', 'move',
        'pick up', 'pickup', 'clean', 'cleaning'
    ]
    
    goal_keywords_count = sum(1 for kw in goal_keywords if kw.lower() in state.lower())
    goal_progress_score = min(1.0, goal_keywords_count / 3)
    
    # Check for location information (closer to goal = higher value)
    location_patterns = [
        r'in the (kitchen|living room|bedroom|bathroom|hallway|dining room|office)',
        r'at the (table|counter|desk|shelves|floor)',
        r'room', 'area', 'location'
    ]
    
    location_found = any(re.search(p, state, re.IGNORECASE) for p in location_patterns)
    location_score = 0.3 if location_found else 0.1
    
    # Check for object state information
    object_patterns = [
        r'cup', 'bottle', 'bowl', 'plate', 'spoon', 'fork', 'knife',
        'phone', 'lamp', 'toilet', 'sink', 'trash', 'box', 'bag',
        'key', 'remote', 'book', 'paper', 'cloth', 'towel'
    ]
    
    objects_mentioned = sum(1 for obj in object_patterns if obj.lower() in state.lower())
    object_score = min(1.0, objects_mentioned / 5)
    
    # Check for positive action indicators
    action_keywords = [
        'move', 'pick', 'drop', 'place', 'clean', 'open', 'close',
        'turn on', 'turn off', 'wash', 'dry'
    ]
    
    action_found = any(kw.lower() in state.lower() for kw in action_keywords)
    action_score = 0.2 if action_found else 0.1
    
    # Calculate composite score
    # Weight factors based on importance
    composite = (
        step_score * 0.3 +
        goal_progress_score * 0.3 +
        location_score * 0.2 +
        object_score * 0.15 +
        action_score * 0.05
    )
    
    # Ensure value is in reasonable range [0, 1]
    return max(0.0, min(1.0, composite))