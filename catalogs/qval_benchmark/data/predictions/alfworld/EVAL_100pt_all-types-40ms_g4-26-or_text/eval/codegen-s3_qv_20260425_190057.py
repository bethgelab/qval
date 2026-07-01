import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and resulting next state in ALFWorld.
    Q(s, a, s') is approximated based on whether the action leads to task completion,
    represents progress, or is a failed/neutral action.
    """
    # Normalize inputs for consistent comparison
    a_low = action.lower().strip()
    n_low = next_state.lower().strip()
    
    # 1. Terminal Success Detection
    # If the next state contains phrases indicating the task is finished, return max reward.
    success_indicators = ["success", "task complete", "goal reached", "finally", "you have successfully"]
    if any(ind in n_low for ind in success_indicators):
        return 1.0
        
    # 2. Failure Detection
    # If the action clearly failed or was invalid, return 0.
    failure_indicators = [
        "nothing happened", 
        "is not here", 
        "is not reachable", 
        "cannot", 
        "does not exist", 
        "not available", 
        "was not found",
        "there is no"
    ]
    if any(ind in n_low for ind in failure_indicators):
        return 0.0

    # 3. Action Effectiveness (Progress) Detection
    # We check if the command was actually executed in the environment.
    is_effective = False
    
    # A. Navigation (e.g., "go to kitchen")
    if a_low.startswith("go to"):
        target = a_low.replace("go to", "").strip()
        # Check if the target room/location is mentioned in the next state description.
        if target in n_low:
            is_effective = True
    
    # B. Manipulation: Pick Up (e.g., "pick up apple")
    elif "pick up" in a_low:
        obj = a_low.replace("pick up", "").strip()
        # Look for evidence that the pick-up action was completed.
        if ("pick" in n_low) and (obj in n_low):
            is_effective = True
            
    # C. Manipulation: Put (e.g., "put apple in fridge")
    elif "put" in a_low and "in" in a_low:
        # Use regex to parse the object and the destination.
        match = re.search(r"put (.*) in (.*)", a_low)
        if match:
            obj, loc = match.groups()
            if obj.strip() in n_low and loc.strip() in n_low:
                is_effective = True
        else:
            # Simple fallback if regex fails: check if the verb and 'in' appear.
            if "put" in n_low and "in" in n_low:
                is_effective = True

    # D. Manipulation: Open or Clean (e.g., "open drawer", "clean table")
    elif "open" in a_low or "clean" in a_low:
        verb = "open" if "open" in a_low else "clean"
        obj = a_low.replace(verb, "").strip()
        # Check for the verb or its past tense and the object being mentioned.
        if (verb in n_low) and (obj in n_low):
            is_effective = True
    
    # E. Generic Fallback
    # If none of the specific patterns match, check if the action's main verb 
    # appears in the next state to see if anything changed.
    if not is_effective:
        # Extract words longer than 3 chars to avoid common small words like 'the', 'to', 'in'.
        words = [w for w in a_low.split() if len(w) > 3]
        for word in words:
            if word in n_low:
                is_effective = True
                break

    # 4. Assign Q-value based on analysis
    if is_effective:
        # A successful step is valued as positive progress toward the goal.
        # Since we don't know the step count, we provide a standard progress value.
        return 0.6
    
    # If the action wasn't a success or a failure, it was likely a neutral 
    # or redundant step (e.g., looking around or an unhelpful command).
    return 0.1