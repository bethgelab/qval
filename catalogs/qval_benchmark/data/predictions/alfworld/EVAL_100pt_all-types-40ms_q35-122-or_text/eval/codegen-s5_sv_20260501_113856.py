def signal_function(state: str) -> float:
    """
    Estimate the state-value for an ALFWorld state.
    
    The value reflects how favorable the current state is for completing the task,
    considering progress indicators, object possession, location relevance, and
    step efficiency.
    """
    state_lower = state.lower()
    
    # Base value for starting or unknown state
    value = 0.15
    
    # Check for task completion (terminal success state)
    if any(term in state_lower for term in ['success', 'completed', 'done', 'finished', 'task complete']):
        return 1.0
    
    # Check if agent is holding the target object (key progress indicator)
    holding_pattern = 'holding' in state_lower
    holding_nothing = any(term in state_lower for term in ['holding nothing', 'holding: nothing', 'you are holding: nothing'])
    
    if holding_pattern and not holding_nothing:
        # Agent has picked up an object - significant progress
        value += 0.40
    
    # Check if agent is at a target-relevant location
    # Common target locations in household tasks
    target_locations = ['kitchen', 'bedroom', 'bathroom', 'living room', 'dining room', 
                        'garbage', 'drawer', 'cabinet', 'shelf', 'table', 'countertop']
    at_target_loc = any(loc in state_lower for loc in target_locations)
    
    if at_target_loc:
        value += 0.15
    
    # Check for action progress indicators (looking in containers, opening things)
    progress_actions = ['look', 'open', 'close', 'put', 'place', 'take', 'find', 'go to']
    has_action_progress = any(action in state_lower for action in progress_actions)
    
    if has_action_progress:
        value += 0.10
    
    # Check if we're near the end of step limit (penalize if too many steps used)
    # Look for step count in state
    import re
    step_match = re.search(r'step\s*(\d+)', state_lower)
    if step_match:
        steps_used = int(step_match.group(1))
        # More steps with same progress = lower value
        step_penalty = min(0.3, steps_used * 0.0075)
        value -= step_penalty
    
    # Check for backtracking or inefficient patterns
    backtracking_indicators = ['back', 'return', 'go back', 'previous', 'again']
    has_backtracking = any(ind in state_lower for ind in backtracking_indicators)
    
    if has_backtracking:
        value -= 0.10
    
    # Check if we're in a dead-end situation (empty room, nothing useful)
    empty_room_indicators = ['nothing here', 'nothing else', 'empty']
    is_empty = any(ind in state_lower for ind in empty_room_indicators)
    
    if is_empty and not holding_pattern:
        value -= 0.05
    
    # Ensure value stays in reasonable bounds
    value = max(0.0, min(1.0, value))
    
    return value