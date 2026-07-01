def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for task completion indicators
    completion_keywords = ['task completed', 'goal achieved', 'success', 'done', 'completed']
    for keyword in completion_keywords:
        if keyword in state_lower:
            return 0.98
    
    # Check for failure or dead-end indicators
    failure_keywords = ['cannot', 'impossible', 'failed', 'error', 'nothing', 'not found']
    for keyword in failure_keywords:
        if keyword in state_lower:
            return 0.05
    
    # Score based on task progress indicators
    progress_score = 0.0
    
    # Object-location relationship indicators (suggesting objects are being moved)
    location_keywords = ['on the', 'in the', 'at the', 'to the', 'from the']
    for keyword in location_keywords:
        if keyword in state_lower:
            progress_score += 0.08
            break
    
    # Agent holding objects (intermediate progress)
    if 'holding' in state_lower or 'have' in state_lower:
        progress_score += 0.12
    
    # Task action indicators
    task_actions = ['put', 'move', 'take', 'clean', 'heat', 'fill', 'open', 'close', 'pick']
    for action in task_actions:
        if action in state_lower:
            progress_score += 0.04
            break
    
    # Object state changes (progress toward goal)
    state_changes = ['clean', 'dirty', 'hot', 'cold', 'open', 'closed', 'filled', 'empty']
    for change in state_changes:
        if change in state_lower:
            progress_score += 0.03
            break
    
    # Estimate based on state information richness
    word_count = len(state.split())
    info_score = min(0.25, word_count / 60)
    
    # Combine scores
    value = progress_score + info_score
    
    # Ensure reasonable bounds for state-value
    return max(0.1, min(0.9, value))