import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a TerminalBench environment.
    Q(s, a) is estimated as the probability of success given the transition.
    """
    # Baseline Q-value: Starting point for an average action
    q_value = 0.4
    
    # 1. Analyze for failure indicators in the next_state
    # Common shell error patterns
    error_patterns = [
        r"command not found",
        r"Permission denied",
        r"No such file or directory",
        r"syntax error",
        r"Invalid argument",
        r"failed",
        r"error:",
        r"FATAL",
        r"Traceback \(most recent call last\)",
        r"cannot open"
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value -= 0.3
            break

    # 2. Analyze for success indicators in the next_state
    success_patterns = [
        r"successfully",
        r"completed",
        r"passed",
        r"correct",
        r"done",
        r"all tests passed",
        r"verification successful"
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.4
            break

    # 3. Analyze the action to see if it's a verification or submission attempt
    # These are high-stakes actions that lead directly to the binary reward.
    verification_actions = [
        r"\bsubmit\b",
        r"verify\.py",
        r"test\.sh",
        r"check\.py",
        r"pytest",
        r"nosetests"
    ]
    
    is_verification = False
    for pattern in verification_actions:
        if re.search(pattern, action, re.IGNORECASE):
            is_verification = True
            break
            
    if is_verification:
        # If it's a verification action and the result doesn't look like an error, it's very promising
        has_error = any(re.search(p, next_state, re.IGNORECASE) for p in error_patterns)
        if not has_error:
            q_value += 0.3
        else:
            q_value -= 0.2

    # 4. Progress indicators: Actions that typically move toward a goal in sysadmin tasks
    progress_actions = [
        r"vim\s", r"nano\s", r"sed\s", r"echo\s", r"cp\s", r"mv\s", r"mkdir\s", r"chmod\s", r"chown\s"
    ]
    for pattern in progress_actions:
        if re.search(pattern, action):
            # Minor boost for taking actionable steps that change the system state
            q_value += 0.05
            break

    # 5. Information gathering actions (neutral but necessary)
    info_actions = [
        r"\bls\b", r"\bpwd\b", r"\bcat\b", r"\bgrep\b", r"\bfind\b", r"\bdf\b", r"\bdu\b", r"\btop\b"
    ]
    for pattern in info_actions:
        if re.search(pattern, action):
            q_value += 0.02
            break

    # Ensure the Q-value is clamped between 0.0 and 1.0
    return max(0.0, min(1.0, q_value))