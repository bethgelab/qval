import re

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value for an action in the OpenApps environment.
    The value is modeled as the expected discounted return, where the 
    maximum possible return is 1.0.
    """
    state_lower = state.lower()
    next_state_lower = next_state.lower()

    # Phase definitions
    success_keywords = [
        "success", "saved", "created", "sent", "added", 
        "confirmed", "completed", "done", "submitted", 
        "deleted", "updated", "modified", "successfully", "finished"
    ]
    submission_keywords = [
        "save", "submit", "send", "confirm", "ok", "post", 
        "finish", "complete", "publish", "upload", "search"
    ]
    entry_keywords = [
        "new", "add", "create", "compose", "plus", "insert", 
        "start", "write", "edit", "form", "details", "title", "description"
    ]
    root_keywords = [
        "home", "dashboard", "app list", "all apps", "main menu", "welcome"
    ]

    # 1. Determine Phase of next_state (Priority: Success > Submission > Entry > Root)
    # This acts as the baseline Q-value (expected future reward)
    base_value = 0.0
    is_success = any(word in next_state_lower for word in success_keywords)
    is_submission = any(word in next_state_lower for word in submission_keywords)
    is_entry = any(word in next_state_lower for word in entry_keywords)
    is_root = any(word in next_state_lower for word in root_keywords)

    if is_success:
        base_value = 1.0
    elif is_submission:
        base_value = 0.7
    elif is_entry:
        base_value = 0.4
    elif is_root:
        base_value = 0.1
    else:
        base_value = 0.0

    # 2. Immediate return for terminal success
    if is_success:
        return 1.0, {"base_value": 1.0, "action_delta": 0.0, "regression_penalty": 0.0, "redundancy_penalty": 0.0}

    # 3. Action Delta (Bonus for productive movements)
    action_delta = 0.0
    
    # Detect transition in phase
    had_submission = any(word in state_lower for word in submission_keywords)
    had_entry = any(word in state_lower for word in entry_keywords)
    had_root = any(word in state_lower for word in root_keywords)

    # Productive Phase Advance
    if is_submission and not had_submission:
        action_delta += 0.1
    elif is_entry and not had_entry and not is_submission:
        action_delta += 0.05

    # Fill productivity
    if action.startswith("fill"):
        match = re.search(r"fill\('.*?',\s*'(.*?)'\)", action)
        fill_text = match.group(1) if match else ""
        if len(fill_text) > 0:
            action_delta += 0.05
            # Bonus for filling in a high-value state
            if is_submission or is_entry:
                action_delta += 0.05
    
    # Scroll productivity
    if action.startswith("scroll"):
        bids_state = re.findall(r'bid=\d+', state)
        bids_next = re.findall(r'bid=\d+', next_state)
        if len(bids_next) > len(bids_state):
            action_delta += 0.05
        elif state == next_state:
            action_delta -= 0.05

    # No-op penalty
    if action.startswith("noop"):
        action_delta -= 0.1

    # 4. Regression Penalty (Moving backwards in the pipeline)
    regression_penalty = 0.0
    if had_submission and is_entry and not is_submission:
        regression_penalty = -0.2
    elif (had_submission or had_entry) and is_root:
        regression_penalty = -0.2

    # 5. Redundancy Penalty (Action had no effect)
    redundancy_penalty = 0.0
    if state == next_state:
        if action.startswith(("click", "press")):
            redundancy_penalty = -0.1

    # Final Summation
    total = base_value + action_delta + regression_penalty + redundancy_penalty
    
    # Constrain total to [0, 1] range to align with binary reward
    total = max(0.0, min(1.0, total))

    return total, {
        "base_value": base_value,
        "action_delta": action_delta,
        "regression_penalty": regression_penalty,
        "redundancy_penalty": redundancy_penalty,
    }