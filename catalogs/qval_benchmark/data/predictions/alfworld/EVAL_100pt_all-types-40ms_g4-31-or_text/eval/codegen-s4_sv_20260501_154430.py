import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment.
    The value is based on the progress toward the goal extracted from the state description.
    """
    state_lower = state.lower()
    
    # Identify the most recent observation to distinguish between the goal description
    # and the current status of the agent and environment.
    if "Observation:" in state:
        current_obs = state.split("Observation:")[-1].lower()
    else:
        current_obs = state_lower

    # Immediate terminal success
    if "successfully" in current_obs or "task completed" in current_obs:
        return 1.0

    # Extract common ALFWorld goals using regex patterns.
    # Pattern 1: "put the [object] in/on/at the [location]"
    put_match = re.search(r"put the (.+?) (in|on|at) the (.+?)(?:\.|\n|$)", state_lower)
    # Pattern 2: "clean the [object]"
    clean_match = re.search(r"clean the (.+?)(?:\.|\n|$)", state_lower)

    if put_match:
        obj = put_match.group(1).strip()
        loc = put_match.group(3).strip()
        
        # Check if the agent is holding the target object
        holding_obj = f"holding the {obj}" in current_obs or f"holding {obj}" in current_obs
        # Check if the agent is at the target location
        at_loc = any(phrase in current_obs for phrase in [f"in the {loc}", f"at the {loc}", f"near the {loc}"])
        
        if holding_obj and at_loc:
            return 0.85  # Nearly complete
        if holding_obj:
            return 0.60  # Key object acquired
        if at_loc:
            return 0.30  # At the destination, but without the object
        if f"{loc} is open" in current_obs:
            return 0.25  # Prerequisite for placing object
        if loc in current_obs or obj in current_obs:
            return 0.15  # Proximity to target items

    if clean_match:
        obj = clean_match.group(1).strip()
        
        # Cleaning requires a tool (usually a cloth or sponge)
        holding_tool = any(tool in current_obs for tool in ["holding the cloth", "holding the sponge", "holding cloth", "holding sponge"])
        at_obj = any(phrase in current_obs for phrase in [f"in the {obj}", f"at the {obj}", f"near the {obj}"])
        
        if holding_tool and at_obj:
            return 0.80  # Tool acquired and at target
        if holding_tool:
            return 0.40  # Tool acquired
        if at_obj:
            return 0.20  # At target, but tool missing
        if obj in current_obs:
            return 0.10  # Proximity to target

    # Generic markers for any task
    if "holding" in current_obs:
        return 0.3
    if "open" in current_obs:
        return 0.1

    return 0.0