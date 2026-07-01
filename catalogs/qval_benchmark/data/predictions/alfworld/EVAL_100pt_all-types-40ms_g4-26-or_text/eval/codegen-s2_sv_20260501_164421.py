import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for an ALFWorld state by parsing the goal 
    and checking for progress indicators in the current observation.
    """
    state_lower = state.lower()
    
    # 1. Terminal Success Check
    # If the episode is already successful, the value is 1.0.
    if any(phrase in state_lower for phrase in ["task completed", "successfully", "you have finished", "goal reached"]):
        return 1.0
        
    # 2. Goal Extraction
    # The goal is typically identified by "Goal:", "Task:", or "Objective:".
    # We assume the goal ends at the first period.
    goal_match = re.search(r"(?:goal|task|objective):\s*([^.]+)", state_lower)
    if not goal_match:
        # If the goal isn't explicitly in the state string, we can't reason about progress.
        return 0.0
        
    goal = goal_match.group(1).strip()
    
    # 3. Scenario Analysis
    # Scenario A: Put/Move/Place/Drop (The agent needs to move an object to a location)
    # Regex matches: (action) (optional 'the') (object) (prep) (optional 'the') (location)
    put_match = re.search(r"(?:put|move|place|drop)\s+(?:the\s+)?(.+?)\s+(?:in|to|on|at)\s+(?:the\s+)?(\w+)", goal)
    if put_match:
        obj = put_match.group(1).strip()
        loc = put_match.group(2).strip()
        
        # Check if the object is already at the destination
        # We look for patterns like "apple in the fridge" or "apple is in the fridge"
        obj_at_loc = False
        for pattern in [rf"{re.escape(obj)}\s+in\s+(?:the\s+)?{re.escape(loc)}",
                        rf"{re.escape(obj)}\s+is\s+in\s+(?:the\s+)?{re.escape(loc)}",
                        rf"{re.escape(obj)}\s+is\s+at\s+(?:the\s+)?{re.escape(loc)}",
                        rf"{re.escape(obj)}\s+at\s+(?:the\s+)?{re.escape(loc)}"]:
            if re.search(pattern, state_lower):
                obj_at_loc = True
                break
        
        # Check if the agent is currently holding the object
        obj_in_hand = any(phrase in state_lower for phrase in [f"holding {obj}", f"carrying {obj}", f"you are holding {obj}"])
        
        # Check if the agent is in the target room
        agent_in_loc = f"you are in the {loc}" in state_lower or f"you are in {loc}" in state_lower
        
        # Check if the object is simply visible in the current observation
        obj_visible = obj in state_lower
        
        if obj_at_loc:
            return 1.0
        if obj_in_hand and agent_in_loc:
            return 0.9
        if obj_in_hand:
            return 0.6
        if obj_visible:
            return 0.3
        return 0.1

    # Scenario B: Clean (The agent needs to change an object's state)
    clean_match = re.search(r"clean\s+(?:the\s+)?(.+)", goal)
    if clean_match:
        obj = clean_match.group(1).strip()
        if f"{obj} is clean" in state_lower:
            return 1.0
        if obj in state_lower:
            return 0.4
        return 0.1
        
    # Scenario C: Find/Get/Pick up (The agent needs to locate or acquire an object)
    find_match = re.search(r"(?:find|get|pick\s+up|grab|take)\s+(?:the\s+)?(.+)", goal)
    if find_match:
        obj = find_match.group(1).strip()
        # If the object is already held, the "get" part of the task is done.
        if any(phrase in state_lower for phrase in [f"holding {obj}", f"carrying {obj}"]):
            return 1.0
        if obj in state_lower:
            return 0.5
        return 0.1
        
    # Fallback for unparsed or complex goals
    return 0.0