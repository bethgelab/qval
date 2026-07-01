def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Convert to lowercase for easier pattern matching
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for immediate success in next state
    success_indicators = ['success', 'completed', 'task completed', 'you have successfully', 
                          'congratulations', 'episode finished', 'done']
    next_state_has_success = any(ind in next_state_lower for ind in success_indicators)
    
    if next_state_has_success:
        return 1.0
    
    # Check for error/invalid action states
    error_indicators = ['error', 'invalid', 'cannot', 'nothing', 'does not exist', 
                        'not here', 'already', 'nothing to', 'you cannot', 'no']
    has_error = any(ind in next_state_lower for ind in error_indicators)
    
    if has_error:
        return -0.1
    
    # Estimate progress based on task completion indicators
    # Look for object-related keywords that indicate progress
    progress_keywords = ['picked up', 'took', 'put', 'placed', 'moved', 'cleaned', 
                         'heated', 'cooled', 'receptacle', 'in the', 'on the', 'to the']
    
    progress_score = 0
    for keyword in progress_keywords:
        if keyword in next_state_lower:
            progress_score += 0.05
    
    # Check if we're closer to goal (compare state and next_state)
    # Count action keywords that typically indicate progress
    progress_actions = ['go to', 'move to', 'walk to', 'take', 'pick up', 'put', 
                        'place', 'clean', 'heat', 'cool', 'open', 'close']
    
    action_progress = 0
    for pa in progress_actions:
        if pa in action_lower:
            action_progress += 0.02
    
    # Penalize redundant or backtracking actions
    redundant_actions = ['look', 'examine', 'wait', 'stop', 'back', 'return']
    redundant_penalty = 0
    for ra in redundant_actions:
        if ra in action_lower:
            redundant_penalty += 0.01
    
    # Check for location changes (indicates movement toward goal)
    location_changed = ('go to' in action_lower or 'move to' in action_lower or 
                        'walk to' in action_lower)
    
    # Base value starts at 0.3 (neutral starting point)
    base_value = 0.3
    
    # Adjust based on progress indicators
    estimated_value = base_value + progress_score + action_progress - redundant_penalty
    
    # Boost value if location changed (moving toward goal)
    if location_changed:
        estimated_value += 0.1
    
    # Check if we have object in hand (important for many tasks)
    if 'holding' in next_state_lower or 'you are holding' in next_state_lower:
        estimated_value += 0.15
    
    # Check if object is at target location
    target_indicators = ['on the', 'in the', 'at the', 'to the']
    if any(ind in next_state_lower for ind in target_indicators):
        estimated_value += 0.1
    
    # Normalize to reasonable range [0, 1]
    estimated_value = max(0.0, min(1.0, estimated_value))
    
    # If we detect task-specific progress (like finding the right object)
    task_keywords = ['microwave', 'fridge', 'sink', 'countertop', 'table', 'drawer', 
                     'cabinet', 'shelf', 'toilet', 'bathtub', 'towel', 'plate', 'bowl',
                     'cup', 'apple', 'banana', 'tomato', 'potato', 'egg', 'knife', 'spoon']
    
    task_progress = 0
    for tk in task_keywords:
        if tk in next_state_lower:
            task_progress += 0.02
    
    estimated_value = min(1.0, estimated_value + task_progress)
    
    return round(estimated_value, 3)