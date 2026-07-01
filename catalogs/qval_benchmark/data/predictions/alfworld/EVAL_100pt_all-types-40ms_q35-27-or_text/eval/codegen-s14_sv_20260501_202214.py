def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for explicit task completion
    completion_signals = ['done', 'completed', 'success', 'finished', 'goal reached', 'task complete', 'congratulations']
    if any(signal in state_lower for signal in completion_signals):
        return 1.0
    
    # Check for explicit failure indicators
    failure_signals = ['fail', 'impossible', 'cannot', 'invalid', 'error', 'not found', 'not accessible']
    if any(signal in state_lower for signal in failure_signals):
        return 0.0
    
    # Count progress indicators
    progress_score = 0.0
    
    # Agent has items (holding, carrying, has)
    if any(word in state_lower for word in ['has ', 'holding', 'carrying', 'inventory']):
        progress_score += 0.2
    
    # Agent location information (in room, at location)
    if any(word in state_lower for word in ['in ', 'at ', 'room', 'location', 'kitchen', 'bedroom', 'bathroom', 'livingroom']):
        progress_score += 0.15
    
    # Object states indicating progress (clean, open, closed)
    if any(word in state_lower for word in ['clean', 'open', 'closed', 'on', 'off']):
        progress_score += 0.15
    
    # Object placement indicators
    if any(word in state_lower for word in ['on the', 'in the', 'at the', 'placed', 'put']):
        progress_score += 0.15
    
    # Goal/target keywords present in state
    if any(word in state_lower for word in ['goal', 'target', 'destination']):
        progress_score += 0.1
    
    # Task action keywords (indicates task is being worked on)
    if any(word in state_lower for word in ['take', 'put', 'move', 'clean', 'open', 'close', 'heat', 'cool']):
        progress_score += 0.1
    
    # Estimate remaining steps based on progress
    # Assume ~15 steps from scratch, reduce based on progress
    estimated_remaining = max(1, 15 - progress_score * 30)
    
    # Apply exponential discounting (gamma = 0.95)
    gamma = 0.95
    discounted_value = gamma ** estimated_remaining
    
    # Cap at 1.0
    return min(1.0, discounted_value)