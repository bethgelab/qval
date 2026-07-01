import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a terminal-based reinforcement learning environment.
    The value represents the expected discounted cumulative reward, which in this 
    binary reward setting is effectively the estimated probability of eventually 
    passing the verifier test script, with a slight preference for efficiency.
    """
    # Baseline Q-value
    q = 0.5

    # 1. Identify Failure Indicators
    # These markers typically indicate that the action did not achieve its intended goal.
    fail_markers = [
        r"command not found",
        r"No such file or directory",
        r"Permission denied",
        r"SyntaxError",
        r"Traceback",
        r"error:",
        r"failed",
        r"Invalid",
        r"cannot access"
    ]
    is_failure = False
    for pattern in fail_markers:
        if re.search(pattern, next_state, re.IGNORECASE):
            is_failure = True
            break

    if is_failure:
        q -= 0.3

    # 2. Identify Success/Progress Indicators
    # These markers suggest the agent is moving toward the goal or has completed a step.
    success_markers = [
        r"Successfully",
        r"Correct",
        r"Done",
        r"completed",
        r"PASSED",
        r"Verification successful",
        r"Installation complete"
    ]
    is_success = False
    for pattern in success_markers:
        if re.search(pattern, next_state, re.IGNORECASE):
            is_success = True
            break

    if is_success:
        q += 0.3

    # 3. Verification Action Analysis
    # In TerminalBench, actions involving 'submit', 'verify', or 'test' are critical.
    # Success on these actions strongly suggests the final reward will be 1.0.
    verify_keywords = ["submit", "verify", "test", "check"]
    if any(kw in action.lower() for kw in verify_keywords):
        if is_success:
            q = 0.95  # High probability of total reward 1.0
        elif is_failure:
            q = 0.1   # Low probability; needs correction
        else:
            q = 0.4   # Neutral/Ambiguous

    # 4. Action-Type Heuristics
    # Productive actions (modifying state) are valued higher than purely exploratory ones,
    # provided they do not result in an error.
    productive_cmds = ["pip install", "gcc", "python", "vim", "nano", "echo", "mkdir", "touch", "sed", "awk"]
    if any(cmd in action for cmd in productive_cmds) and not is_failure:
        q += 0.1

    # Informational commands are useful for planning but don't advance state as much.
    info_cmds = ["ls", "pwd", "cat", "grep", "find", "head", "tail"]
    if any(action.strip().startswith(cmd) for cmd in info_cmds) and not is_failure:
        q += 0.05

    # 5. Final Constraints
    # Ensure the estimated Q-value remains within the valid [0, 1] range.
    return max(0.0, min(1.0, q))