import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the ALFWorld environment.
    The Q-value represents the expected discounted cumulative reward.
    """
    s_low = state.lower()
    a_low = action.lower()
    ns_low = next_state.lower()

    # 1. Immediate Success: Terminal state reward
    if "task completed" in ns_low or "success" in ns_low:
        return 1.0

    # 2. Goal Extraction
    # ALFWorld goals are typically "put the [object] in the [location]" or "clean the [object]"
    target_obj = None
    target_loc = None
    is_clean_goal = False

    # Attempt to find the goal in the current state observation
    put_match = re.search(r"put the (.*?) in the (.*?)(?:\.|\s|$)", s_low)
    if put_match:
        target_obj = put_match.group(1).strip()
        target_loc = put_match.group(2).strip()
    else:
        clean_match = re.search(r"clean the (.*?)(?:\.|\s|$)", s_low)
        if clean_match:
            target_obj = clean_match.group(1).strip()
            is_clean_goal = True

    # 3. Informed Estimation based on extracted goal
    if target_obj:
        if not is_clean_goal:
            # Put goal logic: Target Object and Target Location
            # Action is the final step
            if target_obj in a_low and target_loc in a_low and ("put" in a_low):
                return 0.9
            # Agent is holding the target and is at the target location
            if f"holding {target_obj}" in ns_low:
                if target_loc in ns_low:
                    return 0.8
                return 0.6
            # Agent successfully took the target object
            if "take" in a_low and target_obj in a_low:
                if f"holding {target_obj}" in ns_low:
                    return 0.7
                return 0.4
            # Agent is moving toward the target location
            if "go" in a_low and target_loc in a_low:
                return 0.3
        else:
            # Clean goal logic: Target Object
            if "clean" in a_low and target_obj in a_low:
                return 0.9
            if f"{target_obj} is clean" in ns_low:
                return 0.95
            if "go" in a_low and target_obj in a_low:
                return 0.3

    # 4. General Heuristics (Fallback when goal is not explicitly in state)
    # These capture general progress markers in ALFWorld
    
    # Successful object acquisition
    take_match = re.search(r"take (.*)", a_low)
    if take_match:
        obj = take_match.group(1).strip()
        if f"holding {obj}" in ns_low:
            return 0.6

    # Successful placement or cleaning
    if "is in" in ns_low and "put" in a_low:
        return 0.8
    if "is clean" in ns_low and "clean" in a_low:
        return 0.8
    
    # Holding something is generally better than holding nothing
    if "holding" in ns_low and "holding nothing" not in ns_low:
        if "holding nothing" in s_low: # Just picked something up
            return 0.5
        return 0.4

    # Opening a container to find an object
    if "opened" in ns_low and "closed" in s_low:
        return 0.2

    # Penalty for stagnant states or failed actions
    if "no effect" in ns_low or ns_low == s_low:
        return 0.05
    
    # Baseline for meaningful exploratory actions
    if any(word in a_low for word in ["take", "put", "go", "clean", "open", "examine"]):
        return 0.1

    return 0.0