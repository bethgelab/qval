def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The estimate is based on heuristics identifying progress toward goal completion.
    """
    s_lower = state.lower()
    a_lower = action.lower()
    ns_lower = next_state.lower()

    # 1. Terminal Success
    # ALFWorld usually indicates task completion through explicit markers or final placement.
    if "success" in ns_lower or "task completed" in ns_lower:
        return 1.0

    # 2. Failure or Stagnation
    # If the state didn't change or the environment explicitly denies the action.
    if ns_lower == s_lower:
        return 0.0
    if any(fail in ns_lower for fail in ["cannot", "not possible", "nothing happens", "is not here"]):
        return 0.0

    # 3. High Progress: Object Placement
    # Placing the target object is the final step of most tasks.
    if any(word in a_lower for word in ["put", "place", "drop"]):
        # If the next state indicates the object is now at the destination
        if any(word in ns_lower for word in ["placed", "now in", "is now at"]):
            return 0.9
        return 0.5  # Attempting to place is generally positive

    # 4. Medium-High Progress: Object Acquisition
    # Picking up the target object is a necessary middle step.
    if any(word in a_lower for word in ["take", "pick up", "grab"]):
        if "holding" in ns_lower or "picked up" in ns_lower:
            return 0.7
        return 0.2  # Attempting to take

    # 5. Medium Progress: Cleaning/Preparation
    # Some tasks require cleaning an object before placement.
    if "clean" in a_lower:
        if any(word in ns_lower for word in ["cleaned", "is now clean", "shining"]):
            return 0.6
        return 0.3

    # 6. Low Progress: Navigation
    # Moving to the correct room or object.
    if any(word in a_lower for word in ["go to", "walk to", "move"]):
        # Positive if the observation changed, suggesting a new location or new objects visible.
        if ns_lower != s_lower:
            return 0.2
        return 0.0

    # 7. Other actions (e.g., opening containers)
    if any(word in a_lower for word in ["open", "unlock"]):
        if any(word in ns_lower for word in ["opened", "is now open"]):
            return 0.4

    # Baseline value for generic progress or neutral actions
    return 0.1