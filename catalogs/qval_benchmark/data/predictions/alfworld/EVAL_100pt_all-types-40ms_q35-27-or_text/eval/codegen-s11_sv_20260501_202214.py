def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for goal completion indicators
    if any(word in state_lower for word in ['done', 'success', 'completed', 'goal', 'task']):
        if 'done' in state_lower or 'success' in state_lower:
            return 0.95
        return 0.85
    
    # Check for failure or error indicators
    if any(word in state_lower for word in ['failed', 'cannot', 'impossible', 'error', 'timeout']):
        return 0.05
    
    # Count progress indicators
    progress_indicators = [
        'have', 'holding', 'carrying', 'moved', 'placed', 'put', 'on', 'in',
        'opened', 'closed', 'cleaned', 'heated', 'cooled', 'turned on', 'turned off'
    ]
    progress_count = sum(1 for word in progress_indicators if word in state_lower)
    
    # Count location presence (being in relevant room helps)
    location_indicators = ['kitchen', 'bedroom', 'bathroom', 'livingroom', 'office', 'diningroom', 'closet']
    location_count = sum(1 for word in location_indicators if word in state_lower)
    
    # Count object interaction evidence
    interaction_indicators = ['take', 'open', 'close', 'clean', 'heat', 'cool', 'move', 'pick up', 'put']
    interaction_count = sum(1 for word in interaction_indicators if word in state_lower)
    
    # Count object mentions (more objects mentioned = more task progress)
    object_indicators = ['mug', 'bottle', 'cup', 'bowl', 'plate', 'spoon', 'fork', 'knife',
                         'light', 'switch', 'oven', 'microwave', 'fridge', 'sink', 'toaster']
    object_count = sum(1 for word in object_indicators if word in state_lower)
    
    # Base value starts at 0.1
    base_value = 0.1
    
    # Add progress bonus (up to 0.3)
    progress_bonus = min(progress_count * 0.08, 0.3)
    
    # Add location bonus (up to 0.1)
    location_bonus = min(location_count * 0.03, 0.1)
    
    # Add interaction bonus (up to 0.15)
    interaction_bonus = min(interaction_count * 0.05, 0.15)
    
    # Add object progress bonus (up to 0.1)
    object_bonus = min(object_count * 0.02, 0.1)
    
    # Estimate remaining steps based on progress
    # Assume typical task takes 10-15 steps
    estimated_remaining = 15 - (progress_count + interaction_count + object_count // 2)
    estimated_remaining = max(2, min(15, estimated_remaining))
    
    # Apply discount factor for remaining steps (gamma = 0.95)
    gamma = 0.95
    step_discount = gamma ** estimated_remaining
    
    # Combine all factors
    raw_value = base_value + progress_bonus + location_bonus + interaction_bonus + object_bonus
    value = raw_value * step_discount + 0.05
    
    # Clamp to valid range
    value = max(0.0, min(value, 0.95))
    
    return value