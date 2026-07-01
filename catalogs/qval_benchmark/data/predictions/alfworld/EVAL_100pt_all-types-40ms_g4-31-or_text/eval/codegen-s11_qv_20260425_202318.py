def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given transition in ALFWorld.
    The estimate is based on a heuristic tracking the common sequence of actions:
    Search -> Find -> Pick up -> Move to Target -> Place.
    """
    # Normalize all inputs for case-insensitive matching
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # 1. Success State: Highest value
    # Check for common indicators of task completion
    success_keywords = ["task completed", "successfully", "episode finished", "goal reached", "task finished"]
    if any(keyword in ns for keyword in success_keywords):
        return 1.0

    # 2. Object Placement (Final Step)
    # Taking the action to put an object in its target location
    if "put" in a:
        # If the agent was holding something, this is a high-value attempt
        if "holding" in s:
            # Check if the action actually failed based on the resulting state
            failure_keywords = ["cannot", "fail", "unable", "not possible", "cannot put"]
            if any(fail in ns for fail in failure_keywords):
                return 0.2
            return 0.9
        else:
            # Attempting to put something without holding it is a mistake
            return 0.1

    # 3. Object Acquisition (Early/Mid Stage)
    # Transitions where the agent acquires the target object
    if "holding" in ns:
        if "holding" not in s:
            # Just picked up the object: significant progress
            return 0.7
        else:
            # Already holding the object: maintaining progress
            return 0.6

    # 4. Navigation and Movement
    # Moving towards the goal or the object
    if "go to" in a or "walk" in a or "move" in a:
        # Moving while holding the object is more valuable (approaching target)
        if "holding" in s:
            return 0.5
        # Moving to search for the object
        elif s != ns:
            return 0.3
        else:
            # Moved but stayed in the same state/room
            return 0.1

    # 5. Attempting to interact/take
    # Intent to pick up an object
    if "take" in a or "pick up" in a:
        return 0.4
    
    # 6. Stagnation or Low-Value Actions
    # If state didn't change and action wasn't a purposeful "look" or "examine"
    if s == ns:
        if not any(k in a for k in ["look", "examine", "check"]):
            return 0.05

    # Default value for generic actions or states
    return 0.1