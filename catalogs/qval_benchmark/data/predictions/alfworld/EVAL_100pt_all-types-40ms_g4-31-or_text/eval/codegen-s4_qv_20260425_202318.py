import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an ALFWorld environment.
    The Q-value reflects the expected return, prioritizing progress toward the goal
    and efficiency (shorter trajectories).
    """
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # 1. Terminal Success: The highest possible value.
    success_keywords = ["task completed", "success", "successfully placed", "successfully cleaned"]
    if any(kw in ns for kw in success_keywords):
        return 1.0

    # 2. Object Placement: High value, as it's the final step before success.
    placement_keywords = ["placed the", "put the", "put in"]
    if any(kw in ns for kw in placement_keywords) or (a.startswith("put") and "placed" in ns):
        return 0.95

    # 3. Possession: Holding the target object is a major milestone.
    possession_keywords = ["you have the", "holding the", "you are holding"]
    if any(kw in ns for kw in possession_keywords):
        return 0.8

    # 4. Acquisition: The act of picking up the object.
    acquisition_keywords = ["took the", "picked up the", "picked up"]
    if any(kw in ns for kw in acquisition_keywords) or (a.startswith("take") and "took" in ns):
        return 0.75

    # 5. Discovery: Finding the object or opening the container it is in.
    discovery_keywords = ["you see the", "is inside", "opened the", "found the"]
    if any(kw in ns for kw in discovery_keywords) or (a.startswith("open") and "opened" in ns):
        return 0.5

    # 6. Navigation: Moving to a new location or target area.
    # Check if the action was to go somewhere and the state reflects a location change.
    if a.startswith("go to"):
        # If the next state contains "you are in" or "you are at", navigation was successful.
        if "you are in" in ns or "you are at" in ns:
            # If the state actually changed (different room/location), it's progress.
            if "you are in" in s and "you are in" in ns:
                s_loc = s.split("you are in the")[-1].split(".")[0].strip()
                ns_loc = ns.split("you are in the")[-1].split(".")[0].strip()
                if s_loc != ns_loc:
                    return 0.3
            else:
                return 0.3

    # 7. General Exploration/Interaction: Small positive value for active engagement.
    interaction_keywords = ["examine", "look", "open", "go"]
    if any(kw in a for kw in interaction_keywords):
        return 0.1

    # 8. Baseline: Default value for states/actions that don't show clear progress.
    return 0.0