def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for task completion - highest value
    completion_indicators = [
        'done', 'success', 'completed', 'goal achieved', 'task completed',
        'congratulations', 'you have', 'successfully', 'finished'
    ]
    if any(ind in state_lower for ind in completion_indicators):
        return 1.0
    
    # Check for failure states - lowest value
    failure_indicators = [
        'failed', 'error', 'timeout', 'cannot', 'invalid', 'unable',
        'impossible', 'out of', 'not found', 'does not exist'
    ]
    if any(ind in state_lower for ind in failure_indicators):
        return 0.05
    
    # Base score for valid ongoing state
    score = 0.15
    
    # Check for goal/task description presence
    if 'goal' in state_lower or 'task' in state_lower:
        score += 0.1
    
    # Object in target location patterns (key progress indicator)
    if 'in the' in state_lower or 'on the' in state_lower or 'at the' in state_lower:
        score += 0.1
    
    # Agent has position context
    if 'you are' in state_lower or 'currently' in state_lower or 'you see' in state_lower:
        score += 0.05
    
    # Subtask completion indicators (object state changes)
    subtask_indicators = [
        'opened', 'closed', 'turned on', 'turned off', 'cleaned',
        'heated', 'cooled', 'warmed', 'put', 'placed', 'moved'
    ]
    if any(ind in state_lower for ind in subtask_indicators):
        score += 0.15
    
    # Object manipulation progress (holding/carrying items)
    if any(word in state_lower for word in ['holding', 'carrying', 'have', 'taken', 'picked up', 'grabbed']):
        score += 0.1
    
    # Navigation progress indicators
    if any(word in state_lower for word in ['go to', 'move to', 'walk to', 'navigate', 'go through', 'enter', 'go back']):
        score += 0.05
    
    # Container access progress
    if any(word in state_lower for word in ['inside', 'within', 'container', 'drawer', 'cabinet', 'fridge', 'oven']):
        score += 0.05
    
    # Cap the score to leave room for final completion bonus
    score = min(score, 0.85)
    
    return score