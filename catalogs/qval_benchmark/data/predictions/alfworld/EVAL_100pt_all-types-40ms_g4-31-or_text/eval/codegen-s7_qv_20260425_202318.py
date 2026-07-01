def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The value is based on heuristic milestones: success, possession, discovery, and movement.
    """
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # 1. Success: High reward for completing the task
    success_keywords = ["task completed", "successfully", "has been placed", "is now in", "is now cleaned"]
    if any(kw in ns for kw in success_keywords):
        return 1.0

    # 2. Failure: Low reward for invalid actions or redundant states
    failure_keywords = ["nothing happens", "cannot", "not found", "already", "too far", "invalid"]
    if any(kw in ns for kw in failure_keywords):
        return 0.0

    # 3. Final Action Heuristic: Attempting to place/put an object
    # If the action was to place/put and it didn't fail (checked above), it's highly valuable
    if ("put" in a or "place" in a) and any(kw in ns for kw in ["placed", "put"]):
        return 0.9

    # 4. Possession: Holding the target object is a major milestone
    possession_keywords = ["holding", "carrying", "picked up"]
    holding_now = any(kw in ns for kw in possession_keywords)
    holding_before = any(kw in s for kw in possession_keywords)

    if holding_now:
        if not holding_before:
            return 0.8  # Transition: just acquired the object
        # If already holding, movement toward goal is more valuable than idling
        if any(m in a for m in ["go to", "walk to", "move to"]):
            return 0.7
        return 0.6

    # 5. Discovery: Locating the target object is a minor milestone
    discovery_keywords = ["you see", "is here", "visible", "noticed", "found"]
    discovery_now = any(kw in ns for kw in discovery_keywords)
    discovery_before = any(kw in s for kw in discovery_keywords)

    if discovery_now:
        if not discovery_before:
            return 0.4  # Transition: just found the object
        return 0.3

    # 6. Navigation: Moving to a new room/location
    if ("go to" in a or "walk to" in a) and "you are in" in ns:
        return 0.2

    # 7. Baseline: Default value for non-failure exploration
    return 0.1