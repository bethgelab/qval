import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The estimate is based on identifying progress towards common household goals
    such as moving, cleaning, or picking up objects.
    """
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # 1. Immediate Success Check
    # If the next state indicates the task is finished or completed.
    if any(x in ns for x in ["task completed", "success", "goal reached", "finished", "done"]):
        return 1.0

    # 2. Failure/Invalid Action Check
    # If the action resulted in an error or no change in the environment.
    if any(x in ns for x in ["you cannot", "is not here", "nothing happens", "you are not in", "it is not"]):
        return 0.0

    # 3. Detection of State Transitions (Progress Indicators)
    # We track the "holding" status to identify successful pick-ups or placements.
    is_holding_s = "holding" in s
    is_holding_ns = "holding" in ns
    
    # Navigation detection using regex to find the room name.
    # ALFWorld usually follows the pattern "You are in the [room]."
    navigated = False
    loc_match_s = re.search(r"you are in (?:the )?([\w\s]+?)(?:\.|$)", s)
    loc_match_ns = re.search(r"you are in (?:the )?([\w\s]+?)(?:\.|$)", ns)
    if loc_match_s and loc_match_ns:
        if loc_match_s.group(1).strip() != loc_match_ns.group(1).strip():
            navigated = True

    # 4. Scoring logic based on observed progress
    # Transition: Not holding -> Holding (Pick up)
    if is_holding_ns and not is_holding_s:
        return 0.5
    # Transition: Holding -> Not holding (Drop/Place)
    elif not is_holding_ns and is_holding_s:
        return 0.7
    # Transition: Object becomes clean
    elif "is clean" in ns and "is clean" not in s:
        return 0.6
    # Transition: Object becomes open
    elif "is open" in ns and "is open" not in s:
        return 0.4
    # Transition: Agent changes location
    elif navigated:
        return 0.15
    
    # 5. Fallback: Heuristics based on the action performed (if no state change detected)
    # This helps the agent value "correct" actions that haven't visibly succeeded yet.
    if any(x in a for x in ["go to", "walk", "move", "navigate"]):
        return 0.1
    elif any(x in a for x in ["pick up", "get", "grab"]):
        return 0.2
    elif any(x in a for x in ["put", "place", "drop"]):
        return 0.3
    elif any(x in a for x in ["clean", "wash"]):
        return 0.4
    elif any(x in a for x in ["open"]):
        return 0.2
        
    # Default for uninformative actions like 'look' or repetitive commands.
    if a == "look" or a == "examine" or not a:
        return 0.01
        
    return 0.01