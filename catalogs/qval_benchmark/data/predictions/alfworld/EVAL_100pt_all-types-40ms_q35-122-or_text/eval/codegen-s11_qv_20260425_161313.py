def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Normalize text for comparison
    next_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Check for task completion in next_state
    completion_patterns = [
        'task completed', 'success', 'you have successfully',
        'completed the task', 'congratulations', 'task complete',
        'episode completed', 'task finished', 'completed successfully'
    ]
    
    for pattern in completion_patterns:
        if pattern in next_lower:
            return 1.0
    
    # Check for failure/invalid states
    failure_patterns = [
        'cannot', 'not found', 'does not exist', 'invalid',
        'you cannot', 'there is no', 'nothing to', 'invalid command',
        'cannot go', 'cannot take', 'cannot put', 'not there'
    ]
    
    for pattern in failure_patterns:
        if pattern in next_lower:
            return 0.1
    
    # Analyze action type for productivity
    action_score = 0.0
    
    # Productive actions (object manipulation - highest value)
    productive = ['put', 'place', 'drop', 'take', 'pick up', 'grab', 'get',
                  'clean', 'heat', 'cool', 'open', 'close', 'put in', 'put on',
                  'place on', 'place in', 'give', 'put down']
    for prod in productive:
        if prod in action_lower:
            action_score = 0.3
            break
    
    # Navigation actions (necessary but indirect - medium value)
    nav = ['go to', 'walk to', 'move to', 'goto']
    for n in nav:
        if n in action_lower:
            action_score = 0.1
            break
    
    # Look/examine actions (information gathering - low value)
    info = ['look', 'examine', 'read', 'scan']
    for i in info:
        if i in action_lower:
            action_score = 0.05
            break
    
    # Check for progress indicators in next state (action was executed successfully)
    progress_indicators = [
        'you pick up', 'you take', 'you hold', 'you are holding',
        'you put', 'you place', 'you drop', 'you clean', 'you heat',
        'you cool', 'you open', 'you close', 'you have put', 'you have placed',
        'you have taken', 'you have picked up', 'you have cleaned'
    ]
    
    progress_bonus = 0.0
    for indicator in progress_indicators:
        if indicator in next_lower:
            progress_bonus = 0.2
            break
    
    # Check if we're in a relevant location (task-related areas)
    location_keywords = ['kitchen', 'living room', 'bedroom', 'bathroom',
                         'counter', 'table', 'desk', 'bed', 'drawer',
                         'cabinet', 'fridge', 'microwave', 'sink', 'shelf']
    
    location_bonus = 0.0
    for loc in location_keywords:
        if loc in next_lower:
            location_bonus = 0.05
            break
    
    # Calculate Q-value estimate
    # Base value for valid action execution
    q_value = 0.2 + action_score + progress_bonus + location_bonus
    
    # Penalize if action seems to undo progress (e.g., putting object back)
    undo_patterns = ['put back', 'return to', 'put down', 'drop']
    for undo in undo_patterns:
        if undo in action_lower and 'holding' in state_lower and 'holding' not in next_lower:
            q_value -= 0.1
            break
    
    # Clamp to [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value