import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a transition in the ALFWorld environment.
    
    The Q-value estimates the expected return (1.0 for success, 0.0 for failure).
    Since the reward is sparse, we use heuristics to identify terminal success,
    invalid actions, and progress toward the goal.
    """
    s_lower = state.lower()
    a_lower = action.lower()
    n_lower = next_state.lower()

    # 1. Terminal Success
    # If the next state indicates the task is finished, the return is 1.0.
    success_indicators = [
        'task is complete', 'task completed', 'success', 
        'successfully', 'is now in', 'placed in', 'is in the'
    ]
    if any(ind in n_lower for ind in success_indicators):
        return 1.0

    # 2. Action Failure or Invalid Move
    # If the agent attempts an impossible action or the environment rejects it, 
    # the expected return for that action is effectively 0.0.
    fail_indicators = [
        'you cannot', 'is not possible', 'nothing happens', 
        'invalid', 'does not work', 'no such'
    ]
    if any(ind in n_lower for ind in fail_indicators):
        return 0.0

    # 3. Progress Detection
    # We identify actions that significantly change the environment state.
    
    # A: Change in location
    # Matches "you are in the [room]" where room is a word or phrase.
    loc_regex = r"you are in the ([a-z\s]+?)(?:\.|$)"
    prev_loc_match = re.search(loc_regex, s_lower)
    curr_loc_match = re.search(loc_regex, n_lower)
    if prev_loc_match and curr_loc_match:
        if prev_loc_match.group(1).strip() != curr_loc_match.group(1).strip():
            return 0.7  # Significant progress made by moving

    # B: Change in inventory (picking up/dropping objects)
    # Matches "carrying an [object]" or "carrying [object]".
    inv_regex = r"carrying (?:an? )?([a-z]+)"
    prev_inv_match = re.search(inv_regex, s_lower)
    curr_inv_match = re.search(inv_regex, n_lower)
    
    prev_inv = prev_inv_match.group(1).strip() if prev_inv_match else "nothing"
    curr_inv = curr_inv_match.group(1).strip() if curr_inv_match else "nothing"
    
    # If we transitioned from nothing to an object (or changed objects), it's progress.
    if prev_inv != curr_inv and curr_inv != "nothing":
        return 0.7

    # C: Object state change (e.g., cleaning or washing)
    if any(kw in a_lower for kw in ['clean', 'wash', 'scrub']):
        if any(ind in n_lower for ind in ['is clean', 'is washed', 'is shiny', 'is spotless']):
            return 0.7

    # 4. Neutral / No Progress
    # If the text description remains substantially the same, the action was likely a waste.
    if n_lower.strip() == s_lower.strip():
        return 0.0
    
    # If there was a change in description but no clear progress was detected, 
    # return a low non-zero value to signify a non-failure transition.
    return 0.1