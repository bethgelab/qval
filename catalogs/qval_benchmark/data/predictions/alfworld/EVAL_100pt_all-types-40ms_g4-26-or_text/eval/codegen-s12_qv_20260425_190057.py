import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The estimate is based on task completion, progress towards goal-related 
    objects/locations, and successful execution of actions.
    """
    s_low = state.lower()
    a_low = action.lower()
    ns_low = next_state.lower()

    # 1. Immediate Reward: Check if the task is completed
    # Successful terminal states often contain these patterns
    success_indicators = ["success", "task completed", "goal reached", "all objectives met", "everything is in its place"]
    if any(indicator in ns_low for indicator in success_indicators):
        return 1.0

    # 2. Penalty: Check for invalid or non-progressive actions
    # If the environment indicates nothing happened or the action was invalid
    fail_indicators = ["can't", "nothing happens", "not possible", "is not here", "is not in"]
    if any(indicator in ns_low for indicator in fail_indicators):
        return 0.0
    
    # If the state hasn't changed at all, it's likely a wasted move
    if ns_low.strip() == s_low.strip():
        return 0.0

    # 3. Extract Goal Context
    # Attempt to identify the core components of the goal to reward relevant actions
    goal_keywords = []
    # Common pattern: "Your task is to [move the apple to the fridge]."
    goal_match = re.search(r"task is to (.*?)\.", s_low)
    if goal_match:
        goal_text = goal_match.group(1)
        # Extract words longer than 2 chars to avoid noise (the, to, in, etc.)
        goal_keywords = [w for w in re.findall(r'\w+', goal_text) if len(w) > 2]

    # 4. Parse Action to identify Verb and Targets
    verb = ""
    targets = []
    # List of common ALFWorld verbs
    verbs = ["pick up", "put", "go to", "go", "drop", "clean", "open", "close", "get", "take"]
    
    found_verb = False
    for v in verbs:
        if v in a_low:
            verb = v
            found_verb = True
            # Everything after the verb is treated as a target or set of targets
            target_part = a_low.split(v, 1)[1].strip()
            
            # Clean common filler words from targets
            target_part = re.sub(r'\b(the|a|an|to|in|on|at)\b', '', target_part).strip()
            
            if verb in ["pick up", "get", "take"]:
                targets = [target_part] if target_part else []
            elif verb in ["put", "drop"]:
                # Split "apple in fridge" or "apple on table"
                parts = re.split(r'\s+(?:in|on|at)\s+', target_part)
                targets = [p.strip() for p in parts if p.strip()]
            else:
                # For 'go to', 'clean', etc.
                targets = [target_part] if target_part else []
            break
            
    if not found_verb:
        targets = [w for w in a_low.split() if len(w) > 2]

    # 5. Scoring Logic
    score = 0.1  # Base score for any action that doesn't explicitly fail
    
    # Determine if the action was "successful" in the environment context
    is_successful_transition = False
    if "go" in verb:
        if targets and any(t in ns_low for t in targets):
            is_successful_transition = True
    elif any(v in verb for v in ["pick", "get", "take"]):
        if "holding" in ns_low and any(t in ns_low for t in targets):
            is_successful_transition = True
    elif any(v in verb for v in ["put", "drop"]):
        # Check if target object and location are mentioned in the new state
        if len(targets) >= 1 and any(t in ns_low for t in targets):
            is_successful_transition = True
    elif "clean" in verb:
        if "clean" in ns_low and any(t in ns_low for t in targets):
            is_successful_transition = True
    elif any(v in verb for v in ["open", "close"]):
        if targets and any(t in ns_low for t in targets):
            is_successful_transition = True

    # Calculate how many action targets are relevant to the stated goal
    match_count = 0
    for t in targets:
        if any(kw in t for kw in goal_keywords):
            match_count += 1

    if is_successful_transition:
        # Basic success reward
        score = 0.5
        # Boost score if the action targets match the goal keywords
        if targets:
            relevance = match_count / len(targets)
            score += (0.3 * relevance)
        
        # Extra boost for high-relevance actions (e.g., moving the specific object required)
        if match_count > 0 and len(targets) > 0:
            score += 0.15
    else:
        # If the transition wasn't "successful" by parsing but targets the goal, give a small boost
        if match_count > 0:
            score = 0.3

    # Final value clamping: 1.0 is reserved for true terminal success
    return min(max(score, 0.0), 0.95)