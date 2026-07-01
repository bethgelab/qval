def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Early exit if task is complete
    success_keywords = ['success', 'completed', 'done', 'task complete', 'congratulations']
    for keyword in success_keywords:
        if keyword.lower() in next_state.lower():
            return 1.0
    
    # Extract key features from next_state for Q-value estimation
    next_lower = next_state.lower()
    
    # Feature 1: Is agent holding the target object?
    holding_target = 0.0
    if 'holding' in next_lower or 'you are holding' in next_lower:
        holding_target = 0.3
    
    # Feature 2: Is the agent at a relevant receptacle/location?
    at_location = 0.0
    receptacle_keywords = ['receptacle', 'counter', 'sink', 'microwave', 'fridge', 
                           'table', 'desk', 'cabinet', 'drawer', 'shelf']
    for keyword in receptacle_keywords:
        if keyword in next_lower:
            at_location = 0.2
            break
    
    # Feature 3: Has object been prepared (cleaned, heated, etc.)?
    prepared = 0.0
    preparation_keywords = ['cleaned', 'heated', 'cooled', 'charged', 'ready']
    for keyword in preparation_keywords:
        if keyword in next_lower:
            prepared = 0.25
            break
    
    # Feature 4: Is the agent near the target location?
    near_target = 0.0
    if 'near' in next_lower or 'close' in next_lower:
        near_target = 0.15
    
    # Feature 5: Action type quality
    action_quality = 0.0
    good_actions = ['take', 'put', 'clean', 'heat', 'cool', 'charge', 'go to', 'move to']
    if any(action.lower().startswith(g) for g in good_actions):
        action_quality = 0.1
    
    # Feature 6: Check for negative indicators (backtracking, errors)
    negative_score = 0.0
    negative_keywords = ['cannot', 'nothing', 'empty', 'failed', 'error', 'not found']
    for keyword in negative_keywords:
        if keyword in next_lower:
            negative_score = 0.2
            break
    
    # Feature 7: Progress from state comparison
    progress_bonus = 0.0
    if len(next_state) > len(state):
        progress_bonus = 0.1
    
    # Feature 8: Action efficiency - prefer direct actions
    efficiency_bonus = 0.0
    if len(action.split()) <= 4:
        efficiency_bonus = 0.05
    
    # Combine features into Q-value estimate
    # Base Q-value from accumulated progress
    q_value = holding_target + at_location + prepared + near_target + action_quality
    
    # Apply penalties
    q_value -= negative_score
    
    # Apply bonuses
    q_value += progress_bonus + efficiency_bonus
    
    # Scale to reasonable range (0 to 1)
    q_value = max(0.0, min(1.0, q_value))
    
    # Adjust for remaining step budget (40 steps total)
    # Earlier in episode = more room for error, slightly lower Q
    # Later in episode = higher stakes, but if progressing well, higher Q
    step_estimate = len(state.split()) // 20  # Rough proxy for step count
    if step_estimate > 30:
        q_value *= 1.1  # Late game, if still progressing, higher value
    elif step_estimate > 20:
        q_value *= 1.05
    
    return round(q_value, 4)