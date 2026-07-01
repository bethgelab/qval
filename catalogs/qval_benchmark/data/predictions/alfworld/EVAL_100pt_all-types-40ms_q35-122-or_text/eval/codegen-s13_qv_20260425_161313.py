def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check for task completion in next_state
    completed_indicators = [
        'task completed', 'you have successfully', 'congratulations',
        'success', 'finished', 'done', 'goal reached', 'all done'
    ]
    
    for indicator in completed_indicators:
        if indicator.lower() in next_state.lower():
            return 1.0
    
    # Check for error/failure states
    failure_indicators = [
        'cannot', 'error', 'invalid', 'not found', 'failed',
        'does not exist', 'is not a', 'not a valid'
    ]
    
    failure_count = sum(1 for ind in failure_indicators if ind.lower() in next_state.lower())
    if failure_count >= 2:
        return 0.0
    
    # Estimate progress toward goal
    progress = 0.0
    
    # Valid action that changed state
    if state != next_state:
        progress += 0.15
    
    # Agent is holding an object (task progress indicator)
    holding_patterns = ['holding', 'in your hands', 'you are holding']
    if any(p in next_state.lower() for p in holding_patterns):
        progress += 0.2
    
    # Action involves task-relevant verbs
    task_verbs = ['put', 'place', 'drop', 'take', 'grab', 'pick', 'open', 'close', 'clean', 'heat', 'cool']
    if any(v in action.lower() for v in task_verbs):
        progress += 0.15
    
    # In a task-relevant location
    locations = ['kitchen', 'bedroom', 'living room', 'bathroom', 'office', 'garage', 'hallway']
    if any(loc in next_state.lower() for loc in locations):
        progress += 0.1
    
    # Object interaction indicators (placing objects)
    object_interactions = ['on', 'in', 'under', 'above', 'next to', 'near']
    if any(oi in next_state.lower() for oi in object_interactions):
        progress += 0.1
    
    # Estimate remaining steps (more progress = fewer remaining)
    base_remaining = 30
    remaining = max(5, int(base_remaining * (1 - progress)))
    
    # Apply discount factor
    gamma = 0.9
    discount_factor = gamma ** remaining
    
    # Scale progress to probability estimate (0-1)
    success_prob = min(1.0, progress * 2)
    
    return success_prob * discount_factor