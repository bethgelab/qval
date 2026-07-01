def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The Q-value represents the expected discounted cumulative reward, approximating 
    progress towards the successful completion of the task.
    """
    s_low = state.lower()
    a_low = action.lower()
    ns_low = next_state.lower()

    # 1. Detect failures or invalid actions
    # If the environment explicitly states the action couldn't be performed.
    if any(x in ns_low for x in ["cannot", "fail", "not found", "invalid", "unable", "too far"]):
        return 0.0

    # 2. Detect terminal success
    # Highest value given to actions that complete the task.
    if any(x in ns_low for x in ["success", "completed", "task finished", "done", "successfully placed"]):
        return 1.0

    # 3. Detect placement progress
    # Placing an object is the final step before success.
    if any(x in a_low for x in ["put", "place", "drop"]):
        # If the agent is no longer holding the object, it was successfully placed/dropped.
        if "holding" not in ns_low and "carrying" not in ns_low:
            return 0.9
        else:
            # Redundant or failed placement attempt.
            return 0.1

    # 4. Detect item acquisition and possession
    if "holding" in ns_low or "carrying" in ns_low:
        # Progress: Agent just picked up the item.
        if "holding" not in s_low and "carrying" not in s_low:
            return 0.8
        
        # Redundancy: Agent tried to pick up something they already have.
        if any(x in a_low for x in ["take", "pick up", "grasp"]):
            return 0.1
            
        # Maintenance: Agent still possesses the item while moving or interacting.
        return 0.7

    # 5. Detect attempt to acquire item
    # Action to pick up an object is significant progress.
    if any(x in a_low for x in ["take", "pick up", "grasp"]):
        return 0.6

    # 6. Detect navigation progress
    if any(x in a_low for x in ["go to", "walk to", "move to"]):
        # Attempt to identify if the destination is already the current location.
        # Extract the target location from the command.
        target = ""
        if "to " in a_low:
            target = a_low.split("to ")[-1].strip()
        
        # If the agent is already at the target, the action is redundant.
        if target and target in s_low:
            return 0.1
        
        # Otherwise, navigation is generally progress.
        return 0.2

    # Default base value for any other action.
    return 0.1