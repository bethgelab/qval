def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    q_value = 0.0
    
    # Base value for taking any action
    q_value += 0.1
    
    # Analyze action for productivity
    action_lower = action.lower()
    
    # Productive actions that typically progress toward goals
    productive_actions = ['put', 'take', 'go to', 'open', 'close', 'clean', 'turn on', 'turn off', 'examine', 'move', 'pick up', 'place']
    for prod_action in productive_actions:
        if prod_action in action_lower:
            q_value += 0.15
            break
    
    # Check if action seems invalid or unproductive
    unproductive_indicators = ['nothing', 'cannot', 'empty', 'already', 'nothing there', 'nothing happens']
    for indicator in unproductive_indicators:
        if indicator in action_lower:
            q_value -= 0.2
    
    # Analyze state for progress indicators
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Goal-related progress indicators - new positive states in next_state
    progress_indicators = ['on the', 'in the', 'at the', 'clean', 'turned on', 'turned off', 'opened', 'closed', 'on top', 'in the']
    for indicator in progress_indicators:
        if indicator in next_state_lower and indicator not in state_lower:
            q_value += 0.1
    
    # Check for goal completion indicators
    completion_indicators = ['done', 'completed', 'success', 'goal', 'finished', 'task complete', 'successfully']
    for indicator in completion_indicators:
        if indicator in next_state_lower:
            q_value += 0.5
    
    # Check for failure indicators
    failure_indicators = ['cannot', 'failed', 'impossible', 'nothing', 'empty', 'not found', 'does not exist']
    for indicator in failure_indicators:
        if indicator in next_state_lower:
            q_value -= 0.3
    
    # Check if state changed meaningfully
    if state_lower != next_state_lower:
        q_value += 0.05
    
    # Check for navigation progress (moving to new locations)
    nav_indicators = ['went to', 'moved to', 'is now at', 'arrived']
    for indicator in nav_indicators:
        if indicator in next_state_lower:
            q_value += 0.1
            break
    
    # Check for object manipulation success
    manipulation_success = ['took', 'put', 'placed', 'picked up', 'opened', 'closed', 'cleaned', 'turned on', 'turned off']
    for indicator in manipulation_success:
        if indicator in next_state_lower and indicator not in state_lower:
            q_value += 0.15
            break
    
    # Penalize if next state looks worse than current (regression)
    regression_indicators = ['closed', 'turned off', 'removed', 'taken away']
    for indicator in regression_indicators:
        if indicator in next_state_lower and indicator not in state_lower:
            q_value -= 0.1
    
    # Clamp to reasonable range [0.0, 1.0]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value