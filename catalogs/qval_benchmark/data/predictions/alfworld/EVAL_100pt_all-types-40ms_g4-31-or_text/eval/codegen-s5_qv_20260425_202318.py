def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The Q-value is estimated based on progress markers such as object manipulation,
    navigation, and terminal success conditions.
    """
    # Normalize strings for case-insensitive matching
    state_low = state.lower()
    action_low = action.lower()
    next_state_low = next_state.lower()

    # 1. Terminal Success: Highest priority
    # ALFWorld typically signals completion with "Task completed" or "successfully".
    if "task completed" in next_state_low or "successfully" in next_state_low or "goal reached" in next_state_low:
        return 1.0

    # 2. Goal-directed Action Successes
    # Placing the target object in the goal location is the final step.
    if ("put" in action_low or "place" in action_low) and "is now in" in next_state_low:
        return 0.9

    # Regression: Dropping an object usually sets the agent back.
    if "holding" in state_low and "holding" not in next_state_low:
        return 0.1

    # 3. State Milestones
    # Holding the target object is a major milestone toward completion.
    if "holding" in next_state_low:
        return 0.7

    # Cleaning the object is often a required step before placing it.
    if "clean" in action_low and "clean" in next_state_low:
        return 0.6

    # Opening a container allows access to the target object.
    if "open" in action_low and ("open" in next_state_low or "opened" in next_state_low):
        return 0.5

    # 4. Navigation and Exploration
    # Moving to a new location is productive if it changes the state.
    if "go to" in action_low and state != next_state:
        return 0.3

    # General state change indicates some progress or interaction.
    if state != next_state:
        return 0.2

    # 5. Stagnation
    # Action that results in no state change or a failure to progress.
    return 0.0