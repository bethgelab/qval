def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The Q-value is based on inferred progress toward the goal.
    """
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()

    # Check for terminal success
    # Typical ALFWorld success markers
    if "task completed" in next_state_lower or "success" in next_state_lower:
        return 1.0

    # Detect state features
    was_holding = "holding" in state_lower
    is_holding = "holding" in next_state_lower
    
    # Detect action types
    action_go = "go to" in action_lower or "walk to" in action_lower
    action_pick = "pick up" in action_lower or "take" in action_lower
    action_put = "put" in action_lower or "place" in action_lower
    action_clean = "clean" in action_lower

    # Q-Value Reasoning Logic:
    # We prioritize transitions that bring the agent closer to the final goal:
    # Search -> Find/Pick -> Move to Target -> Place/Success

    # Case 1: Successfully picked up an object
    if not was_holding and is_holding:
        return 0.8

    # Case 2: Currently holding the object
    if is_holding:
        if action_go:
            # Moving toward the destination while holding the object is high progress
            return 0.8
        if action_put:
            # Attempted to place the object, but not a terminal success (handled above)
            # This might be putting it in the wrong place or a failed attempt.
            return 0.4
        # Just holding the object without moving toward goal
        return 0.7

    # Case 3: Not holding the object
    if not is_holding:
        if action_go:
            # Navigating to find the object or the target area
            return 0.4
        if action_clean and "clean" in next_state_lower:
            # Successfully cleaned something (often a prerequisite step)
            return 0.5
        if action_pick:
            # Tried to pick up but failed (since is_holding is False)
            return 0.1
        # Generic actions (examine, open, etc.)
        if "open" in next_state_lower:
            return 0.3
        return 0.2

    return 0.1