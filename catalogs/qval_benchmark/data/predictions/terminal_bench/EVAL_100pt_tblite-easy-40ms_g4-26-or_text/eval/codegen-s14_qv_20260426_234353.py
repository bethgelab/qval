import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The Q-value represents the expected discounted cumulative reward.
    """
    a_low = action.lower()
    ns_low = next_state.lower()
    
    # 1. Error Detection (Immediate penalty)
    # If the action resulted in a common terminal error, the expected reward is 0.
    error_patterns = [
        r"command not found",
        r"no such file or directory",
        r"permission denied",
        r"syntax error",
        r"error:",
        r"failed to",
        r"not found",
        r"invalid argument",
        r"could not find"
    ]
    if any(re.search(p, ns_low) for p in error_patterns):
        return 0.0
        
    # 2. Terminal/Verification Logic
    # We distinguish between regular commands and commands intended to verify the task.
    verification_keywords = ["test", "verify", "check", "submit", "run", "pytest", "unittest", "evaluate"]
    is_verifying = any(kw in a_low for kw in verification_keywords)
    
    # Specific patterns indicating the task has been successfully completed.
    success_patterns = [
        r"test\s+passed",
        r"all\s+tests\s+passed",
        r"0\s+errors",
        r"completed\s+successfully",
        r"passed\s+successfully",
        r"success"
    ]
    
    if is_verifying:
        # If it's a verification command, the presence of success/failure is decisive.
        if any(re.search(p, ns_low) for p in success_patterns):
            return 1.0
        if "fail" in ns_low:
            return 0.0
        # If the command was a verification attempt but the output is ambiguous,
        # we return a low value as it's likely not a clear success yet.
        return 0.2
        
    # 3. Progress Estimation (For non-terminal steps)
    # For intermediate steps, we estimate the Q-value as the probability of eventual success.
    # A command that executes without error is treated as a neutral-to-good step.
    q_value = 0.5
    
    # Boost for common constructive or setup actions.
    constructive_cmds = [
        "mkdir", "pip", "apt", "install", "cp", "mv", "chmod", 
        "python", "gcc", "make", "cd", "cat", "grep", "sed", 
        "nano", "vim", "write", "echo", "create", "wget", "curl",
        "tar", "unzip", "download", "extract"
    ]
    if any(cmd in a_low for cmd in constructive_cmds):
        q_value += 0.15
        
    # If the action produces significant output, it's more likely to be a meaningful step.
    if len(ns_low.strip()) > 50:
        q_value += 0.05
    
    # If the output contains "success" but it's not a verification step, 
    # it's likely just a filename, directory, or string, so we provide a small boost.
    if any(re.search(p, ns_low) for p in success_patterns):
        q_value += 0.1
        
    # If the command is very short (e.g., 'ls'), we are slightly more cautious.
    if len(a_low.strip()) <= 2:
        q_value -= 0.1

    # Ensure the result is clamped within the [0.0, 1.0] range.
    return max(0.0, min(1.0, q_value))