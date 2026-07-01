def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The estimate is based on error detection, success detection, and the utility of the command.
    """
    # Normalize inputs for consistent analysis
    next_lower = next_state.lower()
    action_lower = action.lower().strip()

    # 1. Error Detection: If the command resulted in a shell error, the Q-value is 0.
    # These patterns indicate the action failed to execute as intended.
    error_patterns = [
        "command not found", 
        "no such file", 
        "permission denied",
        "not a directory", 
        "not found", 
        "error:", 
        "invalid option",
        "syntax error", 
        "unrecognized", 
        "fatal error",
        "moduleerror",
        "filenotfounderror"
    ]
    if any(pattern in next_lower for pattern in error_patterns):
        return 0.0

    # 2. Success Detection: If the command output suggests a successful task completion.
    # These patterns indicate the agent has reached a goal or made significant progress.
    success_patterns = [
        "success", 
        "completed", 
        "done", 
        "verified", 
        "ok",
        "all tests passed", 
        "1 file found", 
        "passed"
    ]
    if any(pattern in next_lower for pattern in success_patterns):
        return 1.0

    # 3. Action-based Heuristics: Evaluate potential based on command utility.
    # We categorize commands into productive, investigative, or manipulative tiers.
    
    # Tier 1: Productive/Execution Commands (High Potential)
    # Commands that run code, compile, or perform cryptographic/system tasks.
    productive_cmds = ["python", "gcc", "g++", "make", "openssl", "pip", "apt", "./", "node", "perl"]
    if any(cmd in action_lower for cmd in productive_cmds):
        return 0.6

    # Tier 2: Inspection/Verification Commands (Medium Potential)
    # Commands used to observe the environment or verify the state.
    inspection_cmds = ["grep", "cat", "diff", "find", "stat", "file", "ls", "head", "tail", "wc"]
    if any(cmd in action_lower for cmd in inspection_cmds):
        return 0.4

    # Tier 3: Manipulation/Navigation Commands (Lower-Medium Potential)
    # Commands that change the environment or move through the filesystem.
    manipulation_cmds = ["mkdir", "cp", "mv", "rm", "cd", "touch", "chmod", "chown"]
    if any(cmd in action_lower for cmd in manipulation_cmds):
        return 0.3

    # 4. Default fallback: 
    # If the action resulted in non-empty output and was not an error/success, 
    # it's treated as a low-value informative action.
    if len(next_state.strip()) > 0:
        return 0.2

    # If the action yielded no visible change (e.g., a no-op command), it's valued minimally.
    return 0.1