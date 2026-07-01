import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a terminal-based reinforcement learning environment.
    The Q-value represents the expected discounted cumulative reward (binary 1.0 or 0.0).
    """
    # Start with a neutral baseline probability of success.
    q = 0.5
    
    # Normalize strings for easier analysis
    action_lower = action.lower().strip()
    next_state_lower = next_state.lower()
    
    # 1. Terminal success indicators (High Reward)
    # These are strong signals that the goal is reached or imminent.
    success_patterns = [
        r"success", r"verified", r"completed", r"correct", 
        r"goal reached", r"challenge solved", r"authentication successful"
    ]
    if any(re.search(pattern, next_state_lower) for pattern in success_patterns):
        q += 0.4
        
    # Special case: submitting the answer
    if "submit" in action_lower:
        if any(re.search(pattern, next_state_lower) for pattern in success_patterns):
            return 1.0  # Highly likely to be the terminal success state
        elif "fail" in next_state_lower or "wrong" in next_state_lower or "incorrect" in next_state_lower:
            return 0.0  # Submission failed

    # 2. Terminal failure indicators (Low Reward)
    # These are signals that the action was counter-productive or invalid.
    failure_patterns = {
        r"command not found": -0.4,
        r"no such file or directory": -0.3,
        r"permission denied": -0.3,
        r"invalid option": -0.2,
        r"syntax error": -0.3,
        r"cannot open": -0.2,
        r"failed": -0.2,
        r"error:": -0.1
    }
    for pattern, penalty in failure_patterns.items():
        if re.search(pattern, next_state_lower):
            q += penalty

    # 3. Progress indicators (Incremental Reward)
    # Action leads to informative output without errors.
    informative_commands = ["ls", "pwd", "cat", "grep", "find", "whoami", "id", "netstat", "ps"]
    is_informative_cmd = any(action_lower.startswith(cmd) for cmd in informative_commands)
    
    # If the command was informative and didn't result in an error, and the output changed meaningfully
    if is_informative_cmd and "not found" not in next_state_lower and "error" not in next_state_lower:
        # Check if the output is non-empty (ignoring typical prompt repetitions)
        if len(next_state) > len(state) + 5:
            q += 0.1
            
    # Detection of typical "flags" or secrets in the output (common in TerminalBench tasks)
    # Looking for patterns like CTF{...}, key=..., or long random hex strings
    secret_patterns = [
        r"[a-z0-9]{32,}", # Long hex/alphanumeric strings
        r"flag\s*[:=]\s*", 
        r"key\s*[:=]\s*",
        r"password\s*[:=]\s*"
    ]
    if any(re.search(pattern, next_state_lower) for pattern in secret_patterns):
        q += 0.3

    # 4. Redundancy/Inefficiency Penalty
    # If the next state is almost identical to the current state, the action didn't do much.
    if next_state.strip() == state.strip():
        q -= 0.2

    # Clamp the Q-value between 0.0 and 1.0
    return max(0.0, min(1.0, q))