import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a terminal command action based on its potential 
    to progress toward a solution (success) or lead to a dead end (error).
    """
    # 1. Immediate Success Check (Terminal Reward)
    # If the output of the action indicates the verification script passed.
    success_patterns = [
        r"passed", 
        r"success", 
        r"flag\{", 
        r"correct", 
        r"verification successful",
        r"all tests passed"
    ]
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 1.0

    # 2. Immediate Error Check
    # If the action resulted in a visible error, it's likely a low-value step.
    error_patterns = [
        r"command not found", 
        r"permission denied", 
        r"syntaxerror",
        r"no such file", 
        r"error:", 
        r"exception", 
        r"not found",
        r"bash:.*:", 
        r"failed to",
        r"could not find"
    ]
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.0

    # 3. Heuristic-based Q-value Approximation
    # We start with a neutral base estimate.
    q = 0.5
    
    # Boost for productive-looking commands.
    # Commands used for discovery, inspection, or execution are generally useful.
    useful_cmds = [
        r"ls", r"cat", r"grep", r"find", r"python", r"gcc", r"make", 
        r"sed", r"awk", r"diff", r"strings", r"nm", r"check", r"ls -l"
    ]
    if any(re.search(cmd, action, re.IGNORECASE) for cmd in useful_cmds):
        q += 0.2
        
    # Boost for information gain.
    # In a terminal session, the state usually accumulates history. 
    # An increase in length often signifies new output was provided.
    if len(next_state) > len(state) + 2:
        q += 0.1
    else:
        # A command that produces no output (like 'cd' or a failed command) 
        # is slightly less valuable unless it's a known successful transition.
        q -= 0.1

    # Penalty for command redundancy.
    # If the user repeats the same command, it's likely a loop or lack of progress.
    state_lines = state.strip().split('\n')
    if state_lines:
        last_line = state_lines[-1].strip()
        clean_action = action.strip()
        # Check if the current action was already the most recent thing typed.
        if last_line == clean_action or last_line.endswith(" " + clean_action):
            q -= 0.3

    # Ensure the returned value stays within the [0.0, 1.0] range.
    return max(0.0, min(1.0, q))