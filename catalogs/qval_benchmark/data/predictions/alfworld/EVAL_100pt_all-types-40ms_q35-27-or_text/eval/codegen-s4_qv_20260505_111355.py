def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for task completion in next_state
    success_patterns = ['success', 'completed', 'done', 'finished', 'goal achieved', 'task complete', 'successfully', 'accomplished', 'goal is']
    next_state_lower = next_state.lower()
    
    for pattern in success_patterns:
        if pattern in next_state_lower:
            return 1.0
    
    # Detect if action caused meaningful state change
    state_changed = state != next_state
    
    # Analyze action type
    action_lower = action.lower()
    
    # High-value actions (directly manipulate objects toward goal)
    manipulation_actions = ['take', 'put', 'move', 'clean', 'heat', 'cool', 'open', 'close', 'turn on', 'turn off', 'fill']
    is_manipulation = any(act in action_lower for act in manipulation_actions)
    
    # Medium-value actions (navigation)
    navigation_actions = ['go', 'move to', 'walk to', 'navigate']
    is_navigation = any(act in action_lower for act in navigation_actions)
    
    # Check for goal progress indicators in next state
    progress_indicators = ['has', 'is', 'contains', 'located', 'in the', 'on the', 'at the']
    progress_count = sum(1 for ind in progress_indicators if ind in next_state_lower)
    
    # Check for object state changes
    state_change_keywords = ['is clean', 'is dirty', 'is hot', 'is cold', 'is open', 'is closed', 'is on', 'is off', 'is full', 'is empty']
    state_changes_detected = sum(1 for sc in state_change_keywords if sc in next_state_lower)
    
    # Check for goal-related content
    goal_keywords = ['goal', 'target', 'destination', 'object', 'item', 'container']
    goal_count = sum(1 for kw in goal_keywords if kw in next_state_lower)
    
    # Check for location relevance
    location_keywords = ['room', 'counter', 'table', 'drawer', 'cabinet', 'sink', 'refrigerator', 'microwave', 'stove', 'oven']
    location_count = sum(1 for kw in location_keywords if kw in next_state_lower)
    
    # Base Q-value estimate
    q_value = 0.1
    
    # Reward for state change
    if state_changed:
        q_value += 0.2
    
    # Reward for action type
    if is_manipulation:
        q_value += 0.3
    elif is_navigation:
        q_value += 0.15
    
    # Reward for progress indicators
    q_value += min(progress_count * 0.05, 0.2)
    
    # Reward for object state changes
    q_value += min(state_changes_detected * 0.1, 0.2)
    
    # Reward for goal-related content
    q_value += min(goal_count * 0.05, 0.15)
    
    # Reward for location relevance
    q_value += min(location_count * 0.03, 0.1)
    
    # Cap Q-value at reasonable maximum
    q_value = min(q_value, 0.95)
    
    return q_value