import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given ALFWorld state.
    The value is based on the agent's progress toward the goal, inferred from the text.
    """
    state_lower = state.lower()
    
    # Attempt to separate the goal description from the current observation
    if "observation:" in state_lower:
        parts = state_lower.split("observation:", 1)
        goal_part = parts[0]
        obs_part = parts[1]
    else:
        # Heuristic: assume the first sentence is the goal if it contains typical task verbs
        sentences = state_lower.split(".")
        if sentences and ("put" in sentences[0] or "move" in sentences[0]):
            goal_part = sentences[0]
            obs_part = " ".join(sentences[1:])
        else:
            goal_part = ""
            obs_part = state_lower

    # Extract target object and target location from the goal part
    # Pattern matches "put [a/the] [clean] object in [the] location"
    goal_match = re.search(r"(put|move)\s+(?:a\s+)?(clean\s+)?(.*?)\s+(?:in|to)\s+(?:the\s+)?(.*?)(?:\.|$)", goal_part)
    
    if goal_match:
        is_clean_req = goal_match.group(2) is not None
        target_obj = goal_match.group(3).strip()
        target_loc = goal_match.group(4).strip()
        
        # Analysis of the current observation for progress
        holding = False
        if f"holding {target_obj}" in obs_part or f"holding the {target_obj}" in obs_part:
            holding = True
            
        is_clean = False
        if is_clean_req:
            # Check if the target object is currently described as clean in the observation
            if f"clean {target_obj}" in obs_part:
                is_clean = True
        else:
            # Cleaning not required
            is_clean = True
            
        at_loc = target_loc in obs_part
        
        # Heuristic state-value assignment
        # Order of conditions reflects the expected distance to terminal success (1.0)
        if holding and at_loc and is_clean:
            return 0.9  # Almost there
        if holding and is_clean:
            return 0.6  # Holding the correct object, just needs to move to target
        if holding and at_loc: 
            return 0.5  # At location but object still needs cleaning (or vice versa)
        if holding:
            return 0.4  # Holding the object, needs to clean and move
        if target_obj in obs_part:
            return 0.3  # Object is visible, need to pick it up
        if at_loc:
            return 0.2  # At target location, need to find the object
            
    # General fallback if goal parsing fails or target is not yet encountered
    if "holding" in obs_part:
        return 0.4
    if "see" in obs_part:
        return 0.1
        
    return 0.0