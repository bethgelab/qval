import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal-based environment.
    The Q-value represents the expected discounted reward, which in this case is a binary 
    outcome based on whether a verifier script passes.
    """
    # Baseline Q-value
    q_value = 0.2
    
    # Identify potential error markers in the output
    error_patterns = [
        r"command not found",
        r"Permission denied",
        r"No such file or directory",
        r"SyntaxError",
        r"Traceback \(most recent call last\)",
        r"fatal error",
        r"failed to",
        r"cannot open",
        r"Error: "
    ]
    
    # Identify potential success markers in the output
    success_patterns = [
        r"successfully",
        r"completed",
        r"passed",
        r"correct",
        r"done",
        r"Verification successful",
        r"All tests passed"
    ]

    # Analyze the next_state for errors
    has_error = any(re.search(pattern, next_state, re.IGNORECASE) for pattern in error_patterns)
    # Analyze the next_state for success indicators
    has_success = any(re.search(pattern, next_state, re.IGNORECASE) for pattern in success_patterns)

    # The action 'submit' is the critical terminal step.
    # If submission results in success, Q-value is high.
    if "submit" in action.lower():
        if has_success:
            return 1.0
        if has_error:
            return 0.0
        return 0.5

    # Penalize actions that result in explicit errors
    if has_error:
        q_value -= 0.15
    
    # Reward actions that lead to explicitly positive outcomes
    if has_success:
        q_value += 0.3

    # Heuristic: Reward constructive actions that don't fail
    constructive_commands = [
        r"pip install",
        r"apt-get install",
        r"git clone",
        r"mkdir",
        r"touch",
        r"chmod \+x",
        r"python",
        r"gcc",
        r"make",
        r"vim",
        r"nano",
        r"echo.*>",
        r"cat"
    ]
    
    is_constructive = any(re.search(pattern, action, re.IGNORECASE) for pattern in constructive_commands)
    if is_constructive and not has_error:
        q_value += 0.1

    # Heuristic: Reward discovery/navigation if it doesn't fail
    discovery_commands = [r"ls", r"pwd", r"find", r"grep"]
    is_discovery = any(re.search(pattern, action, re.IGNORECASE) for pattern in discovery_commands)
    if is_discovery and not has_error:
        q_value += 0.05

    # Clamp the Q-value between 0.0 and 1.0
    return max(0.0, min(1.0, q_value))