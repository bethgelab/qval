import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is based on progress towards completing a goal (typically moving or cleaning an object).
    """
    state_lower = state.lower()
    
    # 1. Absolute Success
    # Detect signals that the episode has successfully terminated.
    success_markers = ["task completed", "successfully placed", "successfully cleaned", "goal reached"]
    if any(marker in state_lower for marker in success_markers):
        return 1.0
        
    # 2. Goal Extraction
    # We attempt to identify the target object and target location from the state description.
    # Common ALFWorld patterns: "put the [object] in the [location]" or "clean the [object]".
    target_obj = None
    target_loc = None
    
    put_match = re.search(r"put (?:the )?(\w+) (?:in|on|at) (?:the )?(\w+)", state_lower)
    if put_match:
        target_obj = put_match.group(1)
        target_loc = put_match.group(2)
    else:
        clean_match = re.search(r"clean (?:the )?(\w+)", state_lower)
        if clean_match:
            target_obj = clean_match.group(1)
            # For cleaning tasks, success is often signaled by the object being 'clean'.
            if target_obj and f"{target_obj} is clean" in state_lower:
                return 1.0

    # 3. Progress-based Value Estimation
    # We use a heuristic scale to reward progress:
    # - Holding target and at destination: High value (near completion)
    # - Holding target: Medium-high value
    # - Object located: Medium value
    # - Destination located: Medium-low value
    
    if target_obj:
        # Check if the agent is currently holding the target object
        holding_pattern = rf"(?:holding|carrying) (?:the )?{target_obj}"
        if re.search(holding_pattern, state_lower):
            # If the agent is also at the target destination, success is imminent
            if target_loc and target_loc in state_lower:
                return 0.9
            return 0.7
        
        # Check if the target object is in the current observation (but not yet held)
        if target_obj in state_lower:
            return 0.4
            
    # If target location is known and agent is there, but not holding the object
    if target_loc and target_loc in state_lower:
        return 0.3
        
    # General heuristic: holding any object is better than holding nothing
    if "holding" in state_lower:
        return 0.5
        
    # Default value for starting states or states with no clear progress
    return 0.1