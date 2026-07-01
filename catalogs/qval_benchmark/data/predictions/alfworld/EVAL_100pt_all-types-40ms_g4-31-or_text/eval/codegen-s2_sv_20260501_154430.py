import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is based on progress towards a goal (finding, picking up, 
    cleaning, and placing an object at a target location).
    """
    state_lower = state.lower()
    
    # 1. Terminal Success Check
    # High probability of success if these keywords appear in the state.
    if any(word in state_lower for word in ["success", "completed", "task finished", "successfully"]):
        return 1.0
        
    # 2. Goal Extraction
    # Most ALFWorld goals follow patterns like "put the [object] in the [location]"
    # or "move a [object] to the [location]".
    goal_match = re.search(r"(?:put|move)\s+(?:a|the)\s+(.*?)\s+(?:in|on|at|to)\s+(?:the|a)\s+(.*?)(?:\.|\s|$)", state_lower)
    
    if goal_match:
        target_obj_full = goal_match.group(1).strip()
        target_loc = goal_match.group(2).strip()
        
        # Derive a simplified object name to make matching more robust (e.g., "clean apple" -> "apple")
        obj_name = target_obj_full.replace("clean ", "").replace("the ", "").replace("a ", "").strip()
        needs_cleaning = "clean" in target_obj_full
        
        # Get the most recent observation to analyze current context
        observations = state_lower.split("observation:")
        current_obs = observations[-1] if observations else state_lower
        
        # Feature detection
        is_holding = f"holding {obj_name}" in current_obs or f"holding the {obj_name}" in current_obs
        is_at_loc = target_loc in current_obs
        is_obj_visible = obj_name in current_obs
        is_cleaned = f"cleaned {obj_name}" in current_obs or f"cleaned the {obj_name}" in current_obs
        
        # 3. Value Assignment based on Milestones
        # Case: Agent is holding the object and is at the target location.
        if is_holding and is_at_loc:
            # If the object must be clean, check if it is.
            if needs_cleaning and not is_cleaned:
                return 0.7  # Close, but still needs cleaning.
            return 0.95  # Extremely close to completion.
        
        # Case: Agent is holding the object.
        if is_holding:
            if needs_cleaning and not is_cleaned:
                return 0.5  # Holding it, but cleaning step is pending.
            return 0.8  # Major milestone achieved.
            
        # Case: Object has been cleaned (even if not currently held).
        if needs_cleaning and is_cleaned:
            return 0.6
            
        # Case: Object is visible in the current environment.
        if is_obj_visible:
            return 0.4
            
        # Case: Agent is at the target location (but doesn't have the object).
        if is_at_loc:
            return 0.3
            
    # 4. General Fallbacks
    # If no specific goal was parsed but the agent is manipulating things.
    if "holding" in state_lower:
        return 0.2
        
    # Baseline value for early-stage states.
    return 0.1