import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The estimate is based on heuristic progress markers such as object acquisition, 
    navigation, and goal completion.
    """
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # If the state did not change, the action was likely redundant or ineffective.
    if ns == s:
        return -0.1

    # 1. Immediate Goal Success
    # High value for reaching a terminal successful state.
    if "task completed" in ns or "success" in ns or "successfully" in ns:
        return 1.0

    # 2. Negative Signals (Failures)
    # Penalize actions that result in explicit failure messages.
    failure_words = ["cannot", "not found", "nothing here", "invalid", "not in", "already", "unable"]
    if any(fail in ns for fail in failure_words):
        return -0.2

    # 3. Terminal Interactions
    # Actions that typically finalize the task (putting an object in a place or cleaning).
    # If it didn't trigger "success", it might be a wrong target or a prerequisite step.
    if any(word in a for word in ["put", "place", "clean", "wipe", "drop"]):
        return 0.7

    # 4. Object Manipulation Progress
    is_holding_now = "holding" in ns
    was_holding_before = "holding" in s

    # Successfully picking up an item is a significant jump in progress.
    if is_holding_now and not was_holding_before:
        return 0.6
    
    # If the agent is already holding the required item:
    if is_holding_now:
        # Moving towards a target while holding the object is high value.
        if any(word in a for word in ["go to", "walk", "move"]):
            return 0.5
        # Other actions while holding are moderately valuable.
        return 0.3

    # 5. Exploration and Navigation
    # Moving to different rooms to find items is positive but lower value than manipulation.
    if any(word in a for word in ["go to", "walk", "move"]):
        # Ensure the movement actually happened (not a failure)
        return 0.2

    # Default value for actions that don't clearly advance the state.
    return 0.0