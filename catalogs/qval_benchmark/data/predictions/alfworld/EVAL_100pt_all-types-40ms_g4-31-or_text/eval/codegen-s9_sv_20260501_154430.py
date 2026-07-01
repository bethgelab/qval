import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for an ALFWorld environment state based on progress 
    towards the specified household goal.
    """
    state_lower = state.lower()
    
    # Immediate reward for success
    if "success" in state_lower or "task completed" in state_lower:
        return 1.0
    
    # 1. Extract the goal (target object and target location)
    target_obj = None
    target_loc = None
    
    # Common ALFWorld goal patterns
    goal_patterns = [
        r"put (?:the|a) (.*?) in (?:the|a) (.*?)(?:\.|\s|$)",
        r"put (?:the|a) (.*?) at (?:the|a) (.*?)(?:\.|\s|$)",
        r"take (?:the|a) (.*?) to (?:the|a) (.*?)(?:\.|\s|$)",
        r"clean (?:the|a) (.*?) and put it in (?:the|a) (.*?)(?:\.|\s|$)",
        r"clean (?:the|a) (.*?) and put it at (?:the|a) (.*?)(?:\.|\s|$)"
    ]
    
    for pattern in goal_patterns:
        match = re.search(pattern, state_lower)
        if match:
            target_obj = match.group(1).strip()
            target_loc = match.group(2).strip()
            break
    
    # If no goal can be parsed, we provide a low baseline value
    if not target_obj or not target_loc:
        return 0.1
        
    # 2. Isolate the most recent observation to determine the current agent state
    observations = state_lower.split("observation:")
    latest_obs = observations[-1] if observations else state_lower
    
    # Determine current holding status
    holding_match = re.search(r"you are holding (.*?)(?:\.|\s|$)", latest_obs)
    current_holding = holding_match.group(1).strip() if holding_match else ""
    
    # Determine current location
    loc_match = re.search(r"you are in (?:the|a) (.*?)(?:\.|\s|$)", latest_obs)
    current_loc = loc_match.group(1).strip() if loc_match else ""
    
    # 3. Heuristic Progress Evaluation
    # We assign values based on how close the agent is to the terminal success state.
    
    # Is the agent holding the target object?
    # We check if the target object name is a substring of what the agent is holding.
    is_holding_target = target_obj in current_holding
    
    # Is the agent at the target location or is the target location visible?
    # Check if target_loc is in the current location string or visible in latest observations.
    is_at_target = target_loc in latest_obs or target_loc == current_loc
    
    # Is the target object visible in the environment (but not yet held)?
    is_target_visible = target_obj in latest_obs and not is_holding_target
    
    # Value mapping:
    # - Holding target and at target location: Very high probability of success.
    # - Holding target: Significant progress made.
    # - Target object visible: Target found, next step is manipulation.
    # - At target location: Ready for the final step, but still needs the object.
    # - None of the above: Early stage of the task.
    
    if is_holding_target:
        if is_at_target:
            return 0.85
        else:
            return 0.60
    
    if is_target_visible:
        return 0.40
        
    if is_at_target:
        return 0.20
        
    return 0.10