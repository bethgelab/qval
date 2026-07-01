import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a TerminalBench environment.
    The Q-value represents the expected discounted cumulative reward.
    Since the reward is binary (1.0 on success, 0.0 otherwise), 
    the Q-value is essentially the probability of success given the 
    current action and subsequent optimal play.
    """
    if state is None: state = ""
    if action is None: action = ""
    if next_state is None: next_state = ""

    # 1. Check for terminal submission
    # 'submit' is typically the command used to trigger the verifier.
    if "submit" in action.lower():
        # If the verifier script indicates success, the reward is 1.0.
        success_indicators = ["success", "passed", "correct", "verified", "accepted"]
        if any(ind in next_state.lower() for ind in success_indicators):
            return 1.0
        # If it explicitly fails, the current trajectory is likely dead or needs correction.
        fail_indicators = ["fail", "incorrect", "wrong", "rejected", "error"]
        if any(ind in next_state.lower() for ind in fail_indicators):
            return 0.0
        return 0.1  # Neutral/Uncertain submission result

    # 2. Identify errors in the next state
    # Errors generally indicate a poor action choice.
    error_patterns = [
        r"command not found",
        r"Permission denied",
        r"No such file or directory",
        r"SyntaxError",
        r"Traceback \(most recent call last\)",
        r"fatal error",
        r"invalid option",
    ]
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.1

    # 3. Identify high-value actions (Progress)
    # Actions that modify the environment or execute the primary goal logic.
    high_value_actions = ["python", "gcc", "make", "chmod", "mv", "cp", "echo", "sed", "vim", "nano"]
    is_high_value_action = any(cmd in action for cmd in high_value_actions)

    # 4. Identify information-gathering actions (Discovery)
    # These are necessary but less valuable than direct progress.
    discovery_actions = ["ls", "cat", "pwd", "find", "grep", "whoami", "env", "uname"]
    is_discovery_action = any(cmd in action for cmd in discovery_actions)

    # 5. Analyze state transition for progress
    # Check if the next state contains new, useful information not present in the previous state.
    # If next_state is significantly different and contains a "finding" (like a file list or content), it's good.
    progression_indicators = ["found", "created", "updated", "total", "root:"]
    has_progression = any(ind in next_state.lower() for ind in progression_indicators)

    # Base Q-value calculation
    # We start with a baseline probability of success.
    q_value = 0.3

    if is_high_value_action:
        q_value += 0.3
    elif is_discovery_action:
        q_value += 0.15

    if has_progression:
        q_value += 0.2

    # Penalize redundant actions (taking the same action that doesn't change the state)
    # This encourages efficiency as per the requirement.
    if state == next_state and len(action) > 0:
        q_value -= 0.2

    # Clamp the value between 0.0 and 0.9 (since only 'submit' can reach 1.0)
    return max(0.0, min(0.9, q_value))