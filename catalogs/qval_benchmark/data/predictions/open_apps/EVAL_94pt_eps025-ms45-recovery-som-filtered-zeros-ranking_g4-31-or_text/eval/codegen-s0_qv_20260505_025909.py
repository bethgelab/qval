import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the OpenApps environment.
    Q-value is an estimate of the discounted cumulative reward.
    Since reward is binary (1.0 upon goal completion), Q estimates the 
    likelihood and proximity of achieving the goal.
    """
    # Normalize strings for case-insensitive matching
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()

    # Keywords indicating goal achievement or high probability of success
    success_keywords = [
        "success", "confirmed", "added", "sent", "completed", 
        "created", "saved", "submitted", "deleted", "updated"
    ]
    
    # Keywords indicating progress towards the goal (navigating to forms/actions)
    progress_keywords = [
        "submit", "save", "send", "create", "add", "new", 
        "compose", "edit", "save changes", "confirm"
    ]
    
    # Keywords indicating a setback or error
    error_keywords = [
        "error", "required", "invalid", "failed", "missing", "wrong"
    ]

    # 1. Immediate Success: If the next state contains a success indicator not present in the current state.
    for word in success_keywords:
        if word in next_state_lower and word not in state_lower:
            return 1.0

    # 2. Error Detection: If the action led to an error message.
    for word in error_keywords:
        if word in next_state_lower and word not in state_lower:
            return 0.05

    # 3. Action-based reasoning
    # filling a form is generally productive progress
    if action_lower.startswith("fill"):
        # Check if the fill actually changed something in the state (heuristic)
        if next_state != state:
            return 0.4
        return 0.1

    # Clicking or pressing keys can lead to significant state transitions
    if action_lower.startswith("click") or action_lower.startswith("press"):
        # Check if we transitioned to a state with "Submit/Save" buttons (getting closer to the end)
        for word in progress_keywords:
            if word in next_state_lower and word not in state_lower:
                # Navigating to the final submission step is highly valuable
                if word in ["submit", "save", "send", "confirm"]:
                    return 0.7
                # Navigating to a creation form is moderately valuable
                if word in ["create", "add", "new", "compose"]:
                    return 0.5
        
        # If the state changed but no clear progress keywords appeared, it's still a transition
        if next_state != state:
            return 0.3
        return 0.1

    # Scrolling can be useful if it reveals new interactive elements
    if action_lower.startswith("scroll"):
        if next_state != state:
            # Check if scrolling brought progress keywords into view
            for word in progress_keywords:
                if word in next_state_lower and word not in state_lower:
                    return 0.5
            return 0.2
        return 0.1

    # Default value for noop or unknown actions
    if action_lower.startswith("noop") or next_state == state:
        return 0.1

    return 0.2