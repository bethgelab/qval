import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for an action in an ALFWorld environment based on 
    textual analysis of the state transition.
    """
    s_low = state.lower()
    a_low = action.lower()
    ns_low = next_state.lower()

    # 1. Terminal/Success State: If the task is complete, the return is 1.0
    success_indicators = ["task is complete", "successfully", "goal reached", "you have finished"]
    if any(ind in ns_low for ind in success_indicators):
        return 1.0

    # 2. Failure Detection: If the action resulted in a failure or no change, return 0.0
    failure_indicators = [
        "you cannot", "nothing happens", "not possible", 
        "is not here", "is not reachable", "don't see",
        "is not possible", "are not able"
    ]
    if any(ind in ns_low for ind in failure_indicators):
        return 0.0

    # 3. Progress Detection (Heuristic-based rewards)
    
    # A. Placement Progress (e.g., "put the apple in the fridge")
    # We look for the object and the target being mentioned in the next state.
    put_match = re.search(r"put\s+(?:the\s+)?(\w+)\s+(?:in|into)\s+(?:the\s+)?(\w+)", a_low)
    if put_match:
        obj, target = put_match.groups()
        if obj in ns_low and target in ns_low:
            return 0.8
        elif obj in ns_low:
            # If the object is mentioned but target isn't, it's still potentially progress
            return 0.4

    # B. Manipulation Progress (e.g., "pick up apple", "get the sink")
    pick_match = re.search(r"(?:pick up|get|take|grab)\s+(?:the\s+)?(\w+)", a_low)
    if pick_match:
        obj = pick_match.group(1)
        # Success: "you are holding an apple", "you have the apple"
        if any(p in ns_low for p in [f"holding {obj}", f"have {obj}", f"have an {obj}", f"have a {obj}"]):
            return 0.7
        # If the object is now at the agent's location (even if not explicitly 'held')
        elif obj in ns_low:
            return 0.4

    # C. Navigation Progress (e.g., "go to kitchen")
    nav_match = re.search(r"(?:go to|move to)\s+(?:the\s+)?(\w+)", a_low)
    if nav_match:
        room = nav_match.group(1)
        # Check if the agent is now in the target room
        if any(p in ns_low for p in [f"you are in the {room}", f"you are in a {room}", f"you are in {room}"]):
            return 0.5

    # D. Cleaning Progress (e.g., "clean the sink")
    clean_match = re.search(r"clean\s+(?:the\s+)?(\w+)", a_low)
    if clean_match:
        obj = clean_match.group(1)
        if f"{obj} is clean" in ns_low or f"the {obj} is clean" in ns_low:
            return 0.7

    # 4. Fallback: If the text changed significantly and it wasn't a failure, 
    # it's likely some form of progress or movement.
    if ns_low != s_low:
        return 0.2

    return 0.0