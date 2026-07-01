def signal_function(state: str) -> float:
    """
    Estimate the state-value for an ALFWorld state based on textual analysis.
    
    The state-value represents the expected discounted cumulative reward,
    with higher values indicating states closer to task completion.
    """
    state_lower = state.lower()
    
    # Check for task completion indicators
    completion_keywords = ['done', 'success', 'completed', 'goal achieved', 'task complete', 'success']
    for keyword in completion_keywords:
        if keyword in state_lower:
            return 1.0
    
    # Check for failure indicators
    failure_keywords = ['failed', 'error', 'timeout', 'step limit']
    for keyword in failure_keywords:
        if keyword in state_lower:
            return 0.0
    
    # Analyze state for progress indicators
    score = 0.0
    
    # Count positive progress indicators
    progress_keywords = ['moved', 'placed', 'put', 'cleaned', 'opened', 'closed', 'picked', 'got', 'took']
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    score += min(progress_count * 0.1, 0.3)
    
    # Check if agent is at target location
    location_keywords = ['at the', 'in the', 'on the', 'near the', 'by the']
    if any(kw in state_lower for kw in location_keywords):
        score += 0.1
    
    # Check for object manipulation readiness
    object_keywords = ['table', 'counter', 'drawer', 'cabinet', 'shelf', 'sink', 'fridge', 'microwave']
    object_count = sum(1 for kw in object_keywords if kw in state_lower)
    score += min(object_count * 0.05, 0.2)
    
    # Check for goal-related keywords in state
    goal_keywords = ['goal', 'target', 'destination', 'task', 'need to', 'must']
    if any(kw in state_lower for kw in goal_keywords):
        score += 0.1
    
    # Check for inventory or held items
    inventory_keywords = ['holding', 'carrying', 'have', 'in hand', 'picked up']
    if any(kw in state_lower for kw in inventory_keywords):
        score += 0.1
    
    # Check for navigation capability
    nav_keywords = ['can', 'walk', 'go to', 'move to', 'navigate']
    if any(kw in state_lower for kw in nav_keywords):
        score += 0.05
    
    # Penalize if state seems stuck or has obstacles
    obstacle_keywords = ['cannot', 'blocked', 'locked', 'cannot open', 'cannot take']
    obstacle_count = sum(1 for kw in obstacle_keywords if kw in state_lower)
    score -= min(obstacle_count * 0.1, 0.2)
    
    # Normalize score to [0, 1] range
    score = max(0.0, min(1.0, score))
    
    # Apply step efficiency factor (prefer states that are closer to completion)
    # Assuming we're somewhere in the middle of the episode, scale accordingly
    efficiency_factor = 0.5 + 0.5 * score
    final_value = efficiency_factor
    
    return final_value