import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given transition in a terminal-based RL environment.
    The Q-value represents the expected discounted cumulative reward, which is binary (1.0 on success).
    """
    # Base value: neutral starting point
    q_value = 0.5

    # Patterns that strongly indicate failure
    error_patterns = [
        r"command not found",
        r"no such file or directory",
        r"permission denied",
        r"syntax error",
        r"traceback",
        r"failed to",
        r"error:",
        r"invalid argument",
        r"cannot open"
    ]

    # Patterns that strongly indicate progress or success
    success_patterns = [
        r"successfully",
        r"completed",
        r"finished",
        r"done",
        r"correct",
        r"passed",
        r"verified",
        r"all tests passed",
        r"goal reached"
    ]

    # Check for errors in the transition result
    has_error = False
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            has_error = True
            break
    
    if has_error:
        # Significant penalty for commands that result in errors
        q_value -= 0.4
    else:
        # Bonus for commands that don't error and show signs of success
        for pattern in success_patterns:
            if re.search(pattern, next_state, re.IGNORECASE):
                q_value += 0.3
                break
        
        # If there's no error and no explicit success, but the state changed,
        # it's likely a productive step (e.g., file creation, navigation).
        if next_state != state:
            q_value += 0.1

    # Penalize purely exploratory or redundant actions
    # These are not necessarily 'bad', but less efficient than goal-oriented actions.
    exploratory_actions = ['ls', 'pwd', 'whoami', 'date', 'clear']
    if action.strip().lower() in exploratory_actions:
        q_value -= 0.1

    # High value for the final submission if it appears successful
    if "submit" in action.lower():
        # If submit is called and the output contains success markers, it's likely the final step.
        for pattern in success_patterns:
            if re.search(pattern, next_state, re.IGNORECASE):
                return 1.0
        # If submit is called and fails, it's a very low Q-value.
        if has_error:
            return 0.0

    # Clamp the result between 0.0 and 1.0
    return max(0.0, min(1.0, q_value))