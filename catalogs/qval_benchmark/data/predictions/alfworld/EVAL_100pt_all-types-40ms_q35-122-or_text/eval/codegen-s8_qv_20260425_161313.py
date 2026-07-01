def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Normalize text for analysis
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Extract goal from state
    goal = ""
    goal_patterns = [
        r'your task is to (.+?)(?:\.|\n|$)',
        r'task: (.+?)(?:\.|\n|$)',
        r'goal: (.+?)(?:\.|\n|$)',
        r'put (.+?) on (.+?)(?:\.|\n|$)',
        r'find (.+?) in (.+?)(?:\.|\n|$)',
    ]
    for pattern in goal_patterns:
        match = re.search(pattern, state_lower)
        if match:
            goal = match.group(1).strip()
            break
    
    # Initialize progress score
    progress_score = 0.0
    
    # Check if goal is achieved in next_state
    success_indicators = ['success', 'completed', 'task completed', 'done', 'you have completed']
    if any(ind in next_state_lower for ind in success_indicators):
        return 1.0
    
    # Analyze action type and its impact on progress
    action_type = 'unknown'
    if any(nav in action_lower for nav in ['go to', 'walk to', 'move to', 'navigate']):
        action_type = 'navigation'
        # Navigation can be progress if moving toward target location
        if 'you are now' in next_state_lower or 'you are at' in next_state_lower:
            progress_score += 0.05
    elif any(take in action_lower for take in ['take', 'pick up', 'grab', 'get']):
        action_type = 'pickup'
        if 'you have' in next_state_lower or 'holding' in next_state_lower:
            progress_score += 0.3
    elif any(put in action_lower for put in ['put', 'place', 'drop', 'set']):
        action_type = 'place'
        if 'you have' in next_state_lower or 'on' in next_state_lower:
            progress_score += 0.4
    elif any(clean in action_lower for clean in ['clean', 'wash']):
        action_type = 'clean'
        if 'cleaned' in next_state_lower or 'now clean' in next_state_lower:
            progress_score += 0.2
    elif any(open_close in action_lower for open_close in ['open', 'close', 'look', 'examine']):
        action_type = 'interact'
        progress_score += 0.05
    elif any(invalid in action_lower for invalid in ['nothing', 'cannot', 'no such', 'invalid']):
        action_type = 'invalid'
        progress_score -= 0.2
    
    # Check next_state for progress indicators
    if 'holding' in next_state_lower or 'you have' in next_state_lower:
        progress_score += 0.1
    if 'on' in next_state_lower and 'put' in action_lower:
        progress_score += 0.15
    
    # Penalize if action led to no change or error
    if 'nothing' in next_state_lower and 'take' not in action_lower:
        progress_score -= 0.1
    if 'cannot' in next_state_lower or 'no such' in next_state_lower:
        progress_score -= 0.15
    
    # Estimate remaining steps (heuristic)
    # Typical ALFWorld tasks take 10-20 steps
    remaining_steps = 15.0
    discount_factor = 0.9
    
    # Calculate Q-value: progress * discounted future value
    # Higher progress with fewer remaining steps = higher Q
    step_efficiency = max(0.0, 1.0 - (remaining_steps / 40.0))
    
    q_value = progress_score * step_efficiency
    
    # Normalize to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value + 0.1))
    
    return q_value