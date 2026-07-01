import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The estimate is based on progression towards the household goal (picking up an object, 
    moving to a target location, or completing the task).
    """
    s_lower = state.lower()
    ns_lower = next_state.lower()
    a_lower = action.lower()

    # 1. Check for terminal success (highest value)
    if any(phrase in ns_lower for phrase in ["task completed", "successfully", "goal achieved"]):
        return 1.0
    
    # 2. Attempt to extract the goal (object and location) from the state description
    obj, loc = None, None
    # Common ALFWorld pattern: "Put the X in the Y"
    put_match = re.search(r"put the (.*?) in the (.*?)(?:\.|\n|$)", s_lower)
    # Common ALFWorld pattern: "Clean the X"
    clean_match = re.search(r"clean the (.*?)(?:\.|\n|$)", s_lower)
    
    if put_match:
        obj, loc = put_match.groups()
    elif clean_match:
        obj = clean_match.group(1)
        loc = None  # For cleaning tasks, the object is the destination
    else:
        # Infer goal markers from the action if they aren't explicitly in the current state
        if "put " in a_lower:
            parts = a_lower.split("put ")[1].split(" in ")
            if len(parts) > 1:
                obj, loc = parts[0], parts[1]
        elif "take " in a_lower:
            obj = a_lower.split("take ")[1]
        elif "go to " in a_lower:
            loc = a_lower.split("go to ")[1]

    if obj: obj = obj.strip()
    if loc: loc = loc.strip()

    # 3. Analyze state transitions (Progression markers)
    holding_now = False
    if obj:
        holding_now = (f"holding {obj}" in ns_lower or f"holding the {obj}" in ns_lower)
    
    holding_before = False
    if obj:
        holding_before = (f"holding {obj}" in s_lower or f"holding the {obj}" in s_lower)
        
    at_loc_now = False
    if loc:
        at_loc_now = (f"at the {loc}" in ns_lower or f"in the {loc}" in ns_lower or f"at {loc}" in ns_lower)
    elif obj: 
        # For cleaning tasks, "at the object" is the key progression
        at_loc_now = (f"at the {obj}" in ns_lower or f"at {obj}" in ns_lower)
        
    at_loc_before = False
    if loc:
        at_loc_before = (f"at the {loc}" in s_lower or f"in the {loc}" in s_lower or f"at {loc}" in s_lower)
    elif obj:
        at_loc_before = (f"at the {obj}" in s_lower or f"at {obj}" in s_lower)

    # 4. Heuristic Scoring
    # Priority:- Completion (1.0) -> Ready to complete (0.8) -> Picked up (0.5) -> At location (0.3)
    
    # Case: Agent is holding the object AND is at the target location
    if holding_now and at_loc_now:
        return 0.8
    
    # Case: Action resulted in picking up the target object
    if holding_now and not holding_before:
        return 0.5
    
    # Case: Action resulted in moving to the target location
    if at_loc_now and not at_loc_before:
        return 0.3
    
    # Case: Maintaining a good state (already holding or already at location)
    if holding_now:
        return 0.2
    if at_loc_now:
        return 0.2
    
    # Penalty: Action attempted to put or clean but failed to achieve success/completion
    if ("put " in a_lower or "clean " in a_lower) and not any(phrase in ns_lower for phrase in ["task completed", "successfully"]):
        return 0.0
        
    # Base value for any other valid interaction
    return 0.1