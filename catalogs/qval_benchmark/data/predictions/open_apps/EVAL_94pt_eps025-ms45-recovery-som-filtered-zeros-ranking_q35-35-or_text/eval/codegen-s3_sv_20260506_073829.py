import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for an OpenApps episode based on direct analysis of the state text.
    
    The value is higher if:
    1. The goal appears to be achieved (binary reward 1.0 check passed).
    2. The current state is close to the goal (fewer steps remaining likely).
    3. The state shows progress (e.g., form fields filled, correct page reached).
    4. The state is in a terminal or safe state (no obvious errors).
    
    The value is lower if:
    1. The state is far from the goal (e.g., on home page when goal is complex).
    2. The state shows errors or dead ends.
    3. The state is in a loop or repetitive action without progress.
    """
    
    # Normalize state for analysis
    state_lower = state.lower()
    
    # Check for explicit goal achievement markers
    # These are heuristic patterns that might indicate the goal is met or almost met
    goal_achieved_patterns = [
        r"task\s*completed",
        r"goal\s*achieved",
        r"success",
        r"message\s*sent",
        r"event\s*created",
        r"item\s*added",
        r"calendar\s*updated",
        r"todo\s*completed",
        r"code\s*saved",
        r"map\s*located"
    ]
    
    for pattern in goal_achieved_patterns:
        if re.search(pattern, state_lower):
            return 1.0  # Goal achieved or very close
    
    # Check for error states or dead ends
    error_patterns = [
        r"error",
        r"failed",
        r"not\s*found",
        r"invalid",
        r"timeout",
        r"blocked"
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, state_lower):
            return 0.0  # Dead end or error
    
    # Check for progress indicators
    # Higher value if we are in the middle of a task (e.g., form filled, page loaded)
    progress_patterns = [
        r"form\s*filled",
        r"input\s*entered",
        r"page\s*loaded",
        r"dialog\s*open",
        r"modal\s*open",
        r"button\s*clicked",
        r"link\s*followed"
    ]
    
    progress_score = 0
    for pattern in progress_patterns:
        if re.search(pattern, state_lower):
            progress_score += 0.1
    
    # Check for specific app contexts that might be closer to goal
    # These are heuristic weights based on typical task flows
    app_context_weights = {
        "todo": 0.3,
        "calendar": 0.3,
        "messenger": 0.3,
        "maps": 0.3,
        "code": 0.3,
        "dashboard": 0.2,
        "home": 0.1,
        "login": 0.1
    }
    
    context_score = 0
    for app, weight in app_context_weights.items():
        if app in state_lower:
            context_score = max(context_score, weight)
            break  # Take the most relevant context
    
    # Estimate value based on progress and context
    # Base value starts at 0.0 (no progress)
    estimated_value = 0.0
    
    # Add progress score
    estimated_value += min(progress_score, 0.5)  # Cap progress contribution
    
    # Add context score
    estimated_value += context_score
    
    # If we are in a state that looks like a "next step" is obvious (e.g., form open, button visible)
    # increase value slightly
    if any(re.search(p, state_lower) for p in [r"click.*submit", r"press.*send", r"fill.*text"]):
        estimated_value += 0.1
    
    # Ensure value is between 0.0 and 1.0
    return max(0.0, min(1.0, estimated_value))