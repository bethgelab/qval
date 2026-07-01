def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check if task appears to be completed
    completion_indicators = ['done', 'success', 'completed', 'goal achieved', 'task complete', 'finished']
    if any(indicator in state_lower for indicator in completion_indicators):
        return 1.0
    
    # Check for negative indicators (task failed or stuck)
    failure_indicators = ['failed', 'impossible', 'cannot', 'error', 'timeout', 'limit reached']
    if any(indicator in state_lower for indicator in failure_indicators):
        return 0.0
    
    # Initialize progress score
    progress_score = 0.0
    
    # Agent has target object in hand
    if any(obj in state_lower for obj in ['holding', 'carrying', 'in hand', 'have ']):
        progress_score += 0.35
    
    # Agent is in target location/room
    if any(loc in state_lower for loc in ['target', 'destination', 'goal', 'bedroom', 'bathroom', 'kitchen', 'living room', 'office']):
        progress_score += 0.15
    
    # Object is in correct location (task progress)
    if any(loc in state_lower for loc in ['on ', 'in ', 'at ', 'under ', 'next to ']):
        progress_score += 0.15
    
    # Subtask completion indicators
    if 'cleaned' in state_lower or 'clean ' in state_lower:
        progress_score += 0.1
    if 'opened' in state_lower:
        progress_score += 0.08
    if 'closed' in state_lower:
        progress_score += 0.08
    if 'turned on' in state_lower or 'turned off' in state_lower:
        progress_score += 0.08
    if 'heated' in state_lower or 'cooled' in state_lower:
        progress_score += 0.08
    
    # Navigation progress (agent moved between rooms)
    if any(room in state_lower for room in ['bedroom', 'bathroom', 'kitchen', 'living room', 'office', 'dining room', 'laundry room', 'closet']):
        progress_score += 0.1
    
    # Object manipulation progress
    if any(action in state_lower for action in ['picked up', 'put down', 'moved', 'taken']):
        progress_score += 0.1
    
    # Cap the progress score
    progress_score = min(0.85, progress_score)
    
    # Estimate efficiency factor based on task complexity
    complexity_penalty = 0.0
    if 'clean' in state_lower:
        complexity_penalty += 0.05
    if 'heat' in state_lower or 'cool' in state_lower:
        complexity_penalty += 0.05
    if 'open' in state_lower and 'close' in state_lower:
        complexity_penalty += 0.05
    
    # Combine progress with efficiency consideration
    base_value = progress_score * 0.7
    efficiency_bonus = (0.25 - complexity_penalty)
    
    # Final value estimation
    value = base_value + efficiency_bonus
    
    # Ensure value is in valid range
    return max(0.0, min(1.0, value))