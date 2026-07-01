def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state-action pair in an ALFWorld environment.
    The value is based on the progress toward completing a household task.
    """
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # 1. Check for episode success (highest value)
    success_keywords = ["completed", "success", "finished", "goal reached", "done", "task accomplished"]
    if any(kw in ns for kw in success_keywords):
        return 1.0

    # 2. Check for failure or invalid actions (lowest value)
    failure_keywords = ["can't", "not possible", "nothing happens", "fail", "invalid", "error"]
    if any(kw in ns for kw in failure_keywords):
        return 0.0

    # 3. Putting an object in its target location
    if ("put" in a or "place" in a) and ("put" in ns or "placed" in ns):
        return 0.9

    # 4. Cleaning an object (often a prerequisite)
    if ("clean" in a) and ("clean" in ns or "cleaned" in ns):
        return 0.8

    # 5. Picking up the target object
    if ("take" in a or "pick up" in a) and ("holding" in ns or "picked up" in ns):
        return 0.7

    # 6. Navigation
    if ("go to" in a or "walk to" in a or "move to" in a):
        # If the agent is already holding an object, moving is likely toward the goal
        if "holding" in s or "holding" in ns:
            return 0.5
        # Moving to find/acquire an object
        if s != ns:
            return 0.4
        else:
            return 0.1

    # 7. Exploration and Observation
    if any(kw in a for kw in ["look", "examine", "search", "check"]):
        # Observation is slightly valuable if it leads to a state change or discovery
        if s != ns:
            return 0.2
        return 0.1

    # 8. Default baseline for other interactions that don't fail
    if s != ns:
        return 0.15
    
    return 0.05