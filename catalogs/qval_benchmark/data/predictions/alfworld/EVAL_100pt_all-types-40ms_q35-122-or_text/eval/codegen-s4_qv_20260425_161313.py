def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value starts at neutral
    q_value = 0.0
    
    # Normalize text for analysis
    action_lower = action.lower().strip()
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Extract goal-related information from state
    goal_patterns = [
        r'put\s+\w+\s+(?:in|on|into|onto)\s+\w+',
        r'clean\s+\w+',
        r'heat\s+\w+',
        r'cool\s+\w+',
        r'find\s+\w+',
        r'locate\s+\w+'
    ]
    
    goal_found = any(re.search(p, state_lower) for p in goal_patterns)
    
    # Classify action type and assign base value
    if action_lower.startswith('go to ') or action_lower.startswith('goto '):
        # Navigation - necessary but indirect progress
        q_value = 0.15
    elif action_lower.startswith('take ') or action_lower.startswith('grab '):
        # Picking up object - direct progress toward manipulation
        q_value = 0.35
    elif action_lower.startswith('put ') or action_lower.startswith('place '):
        # Placing object - highly productive, often final step
        q_value = 0.65
    elif action_lower.startswith('clean '):
        # Cleaning action - state change toward goal
        q_value = 0.5
    elif action_lower.startswith('heat '):
        # Heating action - state change toward goal
        q_value = 0.5
    elif action_lower.startswith('cool '):
        # Cooling action - state change toward goal
        q_value = 0.5
    elif action_lower.startswith('open ') or action_lower.startswith('close '):
        # Opening/closing containers - necessary intermediate step
        q_value = 0.25
    elif action_lower.startswith('look'):
        # Looking - informational, minimal direct value
        q_value = 0.05
    elif action_lower.startswith('inventory'):
        # Checking inventory - informational
        q_value = 0.05
    elif action_lower.startswith('break') or action_lower.startswith('destroy'):
        # Destructive actions - usually not helpful
        q_value = -0.2
    else:
        # Unknown or invalid action
        q_value = 0.0
    
    # Check for success indicators in next_state
    success_indicators = ['success', 'completed', 'done', 'task', 'finished', 'congratulations']
    if any(ind in next_state_lower for ind in success_indicators):
        q_value = 1.0
        return q_value
    
    # Check for failure indicators
    failure_indicators = ['failed', 'error', 'invalid', 'cannot', 'not found', 'not here']
    if any(ind in next_state_lower for ind in failure_indicators):
        q_value = max(q_value, -0.3)
    
    # Check if action mentions relevant objects
    if goal_found:
        # Extract potential object from action
        action_words = action_lower.split()
        if len(action_words) > 1:
            target_obj = action_words[1] if action_words[1] not in ['to', 'in', 'on', 'the', 'a'] else None
            if target_obj and target_obj in state_lower:
                q_value += 0.1
    
    # Check if next_state shows progress (object in new location)
    if 'on' in next_state_lower or 'in' in next_state_lower:
        if 'on' in action_lower or 'in' in action_lower:
            q_value += 0.15
    
    # Check for room transitions (progress indicator)
    room_words = ['kitchen', 'bedroom', 'bathroom', 'livingroom', 'office', 'garage', 'basement', 'diningroom']
    if any(room in next_state_lower for room in room_words):
        if not any(room in state_lower for room in room_words):
            q_value += 0.1
    
    # Penalize repetitive actions (same action twice in a row would be wasteful)
    if action_lower in state_lower:
        q_value -= 0.1
    
    # Check if we're near goal completion (fewer steps needed)
    # If next_state mentions the target location, we're close
    if goal_found:
        put_pattern = r'put\s+\w+\s+(?:in|on)\s+(\w+)'
        match = re.search(put_pattern, state_lower)
        if match:
            target_location = match.group(1)
            if target_location in next_state_lower:
                q_value += 0.2
    
    # Ensure Q-value stays in reasonable range
    q_value = max(-0.5, min(1.0, q_value))
    
    return q_value