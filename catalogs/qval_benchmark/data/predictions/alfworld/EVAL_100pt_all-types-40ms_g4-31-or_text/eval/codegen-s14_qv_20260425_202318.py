import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The Q-value is approximated based on the progression toward the task goal.
    """
    s_low = state.lower()
    a_low = action.lower()
    ns_low = next_state.lower()

    # 1. Immediate Success
    # Indicators that the task has been completed.
    success_keywords = ["successfully", "task completed", "goal reached", "has been put", "is now clean", "done"]
    if any(kw in ns_low for kw in success_keywords):
        return 1.0

    # 2. Action Failure
    # Indicators that the action was invalid or did not change the state favorably.
    fail_keywords = ["cannot", "don't see", "not found", "invalid", "not possible", "unable", "i can't", "nothing happens"]
    if any(kw in ns_low for kw in fail_keywords):
        return 0.0

    # 3. Goal Extraction and Milestone-based Estimation
    # ALFWorld goals often follow specific patterns like "put the X in the Y [in the Z]" or "clean the X".
    
    # Case A: Manipulation tasks (Put/Place)
    put_match = re.search(r"(?:put|place) the (.+?) in the (.+?)(?: in the (.+?))?", s_low)
    if put_match:
        item = put_match.group(1).strip()
        container = put_match.group(2).strip()
        room = put_match.group(3).strip() if put_match.group(3) else ""

        # Milestone: Agent is carrying the target item and is in the destination room.
        if (f"carrying {item}" in ns_low or f"holding {item}" in ns_low) and (room and room in ns_low):
            return 0.8
        
        # Milestone: Agent has successfully picked up the target item.
        if f"carrying {item}" in ns_low or f"holding {item}" in ns_low:
            return 0.5
            
        # Milestone: Target item is found in the current observation.
        if item in ns_low:
            return 0.3
            
        # Milestone: Agent has reached the target room.
        if room and room in ns_low:
            return 0.2
            
        # Milestone: Target container is found in the current observation.
        if container in ns_low:
            return 0.1

    # Case B: Cleaning tasks
    clean_match = re.search(r"clean the (.+?)", s_low)
    if clean_match:
        target = clean_match.group(1).strip()
        if target in ns_low:
            # Higher value if the agent is actively performing a cleaning action on the target.
            if any(act in a_low for act in ["clean", "wipe", "scrub"]):
                return 0.6
            return 0.3

    # Baseline for a valid action that doesn't explicitly fail but doesn't reach a milestone.
    return 0.1