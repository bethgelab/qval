import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is based on progress toward the goal: identifying the target object,
    finding it, acquiring it, cleaning it (if required), and reaching the destination.
    """
    state_lower = state.lower()
    
    # 1. Terminal Success Check
    if "task completed" in state_lower or "successfully" in state_lower:
        return 1.0

    # 2. Extract target object and target location from the state
    # Pattern 1: "Put a [object] in the [location]"
    match1 = re.search(r"put\s+(?:a|an)\s+(.+?)\s+in\s+(?:the|a)\s+(.+?)(?:\.|$)", state, re.IGNORECASE)
    # Pattern 2: "Clean a [object] and put it in the [location]"
    match2 = re.search(r"clean\s+(?:a|an)\s+(.+?)\s+and\s+put\s+it\s+in\s+(?:the|a)\s+(.+?)(?:\.|$)", state, re.IGNORECASE)

    if match1:
        obj, loc = match1.groups()
        is_cleaning_task = False
    elif match2:
        obj, loc = match2.groups()
        is_cleaning_task = True
    else:
        # If the goal is not explicitly parsed, return a low baseline value.
        return 0.1

    obj = obj.strip().lower()
    loc = loc.strip().lower()

    # 3. Evaluate Progress
    score = 0.0
    
    # Check if the agent is holding the target object
    holding = False
    if f"holding {obj}" in state_lower or f"carrying {obj}" in state_lower:
        holding = True
        score += 0.4

    # Check if the object is cleaned (required for cleaning tasks)
    cleaned = False
    if is_cleaning_task:
        # Look for indicators that the object has been cleaned
        if f"cleaned {obj}" in state_lower or (holding and "cleaned" in state_lower):
            cleaned = True
            score += 0.3
    else:
        # For non-cleaning tasks, we assume it's "clean enough"
        cleaned = True

    # Check if the agent is at the target location
    # "See a [loc]" or "in the [loc]" are strong indicators of being at the destination.
    at_destination = False
    if f"see {loc}" in state_lower or f"at the {loc}" in state_lower or f"in the {loc}" in state_lower:
        at_destination = True
        if holding and cleaned:
            score += 0.2  # Agent is at the destination with the cleaned object
        elif holding:
            score += 0.1  # Agent is at destination but object isn't cleaned
        else:
            score += 0.05 # Agent is at destination but doesn't have the object

    # Check if the agent has found the object but isn't holding it yet
    if not holding and f"see {obj}" in state_lower:
        score += 0.2

    # 4. Final Value Calculation
    # Ensure the value reflects a discounted cumulative reward.
    # We cap it at 0.95 to leave room for the terminal success value of 1.0.
    # We start from a baseline of 0.05 for valid goals.
    final_val = 0.05 + score
    return min(0.95, final_val)