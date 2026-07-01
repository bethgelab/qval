import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Normalize strings for case-insensitive analysis
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # 1. Check for Terminal Success
    # In ALFWorld, successful completion is explicitly signaled in the observation.
    if 'success' in next_state_lower or 'task complete' in next_state_lower:
        return 1.0
    
    # 2. Check for Terminal Failure / Invalid Action
    if 'error' in next_state_lower or 'invalid' in next_state_lower or 'cannot' in next_state_lower:
        return -1.0
    
    # 3. Check for Stagnation
    # If the state does not change, the action was ineffective or invalid.
    if state == next_state:
        return -0.5
    
    # 4. Analyze Action Quality
    # Manipulation actions (put, clean) are generally higher value than navigation (go).
    action_score = 0.0
    if 'put' in action_lower or 'clean' in action_lower:
        action_score = 0.4
    elif 'take' in action_lower:
        action_score = 0.2
    elif 'go' in action_lower or 'move' in action_lower:
        action_score = 0.1
    elif 'look' in action_lower:
        action_score = 0.05
    else:
        action_score = 0.0
        
    # 5. Analyze State Progress
    # Count location prepositions to detect object placement progress.
    # Regex for common location indicators.
    location_pattern = r'\b(on|in|at|to)\b'
    state_loc_count = len(re.findall(location_pattern, state_lower))
    next_loc_count = len(re.findall(location_pattern, next_state_lower))
    
    progress_score = 0.0
    if next_loc_count > state_loc_count:
        progress_score = 0.2
    elif next_loc_count == state_loc_count and state != next_state:
        # State changed but location count didn't increase (e.g. object moved)
        # Still counts as progress since state != next_state
        progress_score = 0.1
    else:
        progress_score = 0.0
        
    # 6. Combine and Clamp
    # Q-value approximation based on immediate progress and action quality.
    estimated_q = action_score + progress_score
    
    # Clamp to [-1.0, 1.0] range
    return max(-1.0, min(1.0, estimated_q))