def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check for task completion indicators
    completion_patterns = [
        r'(goal|task|episode)\s*(completed|finished|done|success)',
        r'you\s+have\s+completed',
        r'episode\s+ended\s+successfully'
    ]
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 1.0
    
    q_value = 0.0
    
    # State changed indicates action had an effect
    if state != next_state:
        q_value += 0.2
        
        # Check for positive progress indicators in next_state
        positive_indicators = [
            'picked up', 'placed', 'put in', 'put on', 'cleaned',
            'turned on', 'turned off', 'opened', 'closed', 'moved to',
            'in your inventory', 'in the', 'on the', 'took', 'moved'
        ]
        for indicator in positive_indicators:
            if indicator in next_state.lower():
                q_value += 0.1
                break
    
    # Check if action is task-relevant
    action_lower = action.lower()
    relevant_actions = ['take', 'put', 'move', 'clean', 'turn on', 'turn off', 
                       'open', 'close', 'go to', 'navigate to', 'go', 'walk to']
    if any(act in action_lower for act in relevant_actions):
        q_value += 0.1
    
    # Check for goal information in state (indicates task is understood)
    if 'goal' in state.lower() or 'task' in state.lower():
        q_value += 0.05
    
    # Check for object-goal alignment in next_state
    goal_objects = ['in the', 'on the', 'at the']
    for obj in goal_objects:
        if obj in next_state.lower():
            q_value += 0.05
            break
    
    # Penalize for negative feedback or no progress
    negative_indicators = ['cannot', 'unable', 'nothing', 'empty', 'error', 
                          'failed', 'not found', 'does not exist', 'nothing there']
    if any(ind in next_state.lower() for ind in negative_indicators):
        q_value = max(0.0, q_value - 0.15)
    
    # Ensure value is in reasonable range [0, 1]
    q_value = max(0.0, min(q_value, 1.0))
    
    return q_value