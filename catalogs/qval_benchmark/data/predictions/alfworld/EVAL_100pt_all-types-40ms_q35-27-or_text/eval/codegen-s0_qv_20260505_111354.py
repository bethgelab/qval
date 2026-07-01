def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for task completion in next state
    completion_indicators = ['goal', 'achieved', 'success', 'completed', 'done', 'finish']
    next_lower = next_state.lower()
    if any(ind in next_lower for ind in completion_indicators):
        return 1.0
    
    # Check for action errors
    error_indicators = ['cannot', 'invalid', 'error', 'nothing happens', 'no', 'already', 'not possible', 'fail']
    has_error = any(err in next_lower for err in error_indicators)
    
    # Check for progress
    progress_indicators = ['took', 'picked', 'moved', 'opened', 'closed', 'turned', 'put', 'placed', 'cleaned', 'washed', 'filled', 'heated', 'cooled']
    has_progress = any(prog in next_lower for prog in progress_indicators)
    
    # Analyze action type
    action_lower = action.lower()
    is_manipulation = any(act in action_lower for act in ['take', 'put', 'open', 'close', 'turn', 'clean', 'wash', 'fill', 'heat', 'cool'])
    is_navigation = 'go' in action_lower or 'walk' in action_lower
    
    # Base Q-value
    q_value = 0.0
    
    # Penalize errors heavily
    if has_error:
        q_value = -0.5
    
    # Reward progress
    if has_progress:
        q_value += 0.4
    
    # Reward meaningful manipulation actions
    if is_manipulation and not has_error:
        q_value += 0.2
    
    # Slight penalty for pure navigation (less directly productive)
    if is_navigation and not has_progress:
        q_value -= 0.1
    
    # Check if state contains goal-related objects
    goal_keywords = ['mug', 'bowl', 'plate', 'cup', 'spoon', 'fork', 'knife', 'bottle', 'glass']
    state_lower = state.lower()
    if any(obj in state_lower for obj in goal_keywords):
        q_value += 0.1
    
    # Return bounded Q-value
    return max(-1.0, min(1.0, q_value))