import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value of taking 'action' in 'state' resulting in 'next_state'.
    The estimation is based on immediate success/failure indicators and the nature of the action.
    """
    # Normalize inputs to lowercase for robust keyword matching
    ns = next_state.lower()
    a = action.lower()

    # 1. Immediate Success Check
    # We prioritize specific success indicators.
    # The order and logic ensure that "0 errors, 0 failures" is treated as success,
    # while "failed" or "not passed" triggers the failure logic.
    success_indicators = [
        "0 errors, 0 failures", 
        "test passed", 
        "all tests passed", 
        "success", 
        "passed", 
        "ok", 
        "done", 
        "completed", 
        "verified"
    ]
    
    is_success = False
    for indicator in success_indicators:
        if indicator in ns:
            # Check for negative qualifiers that might negate the indicator
            if "not ok" in ns or "not passed" in ns or "failed" in ns:
                # Special case: "0 errors, 0 failures" is explicitly success
                if "0 errors, 0 failures" in ns:
                    is_success = True
                else:
                    is_success = False
                    break
            else:
                is_success = True
                break
    
    if is_success:
        return 1.0

    # 2. Immediate Failure Check
    # If the state contains failure keywords, we return 0.0.
    failure_indicators = [
        "error", 
        "fail", 
        "not found", 
        "denied", 
        "no such", 
        "command not found", 
        "syntax error", 
        "segmentation fault", 
        "exit status 1", 
        "permission denied",
        "invalid"
    ]
    for indicator in failure_indicators:
        if indicator in ns:
            return 0.0

    # 3. Verification/Submission Attempt
    # If the action was intended to verify the solution but no success was detected,
    # the expected reward is 0.0.
    verification_keywords = ["verify", "test", "check", "submit", "pytest", "unittest", "./verify"]
    if any(kw in a for kw in verification_keywords):
        return 0.0

    # 4. Progress Estimation
    # If no immediate success/failure is found, we estimate based on the action type.
    # Productive actions: Actions that modify the system or install dependencies.
    productive_keywords = [
        "install", "mkdir", "cp", "mv", "chmod", "chown", "pip", "apt", 
        "python", "gcc", "make", "sed", "nano", "vim", "vi", "git", 
        "curl", "wget", "echo", "cat", "write", "create", "edit", "touch"
    ]
    # Informational actions: Actions that probe or navigate the environment.
    info_keywords = ["ls", "cd", "pwd", "find", "grep", "whoami", "dir", "stat", "type"]

    if any(kw in a for kw in productive_keywords):
        # Moving towards a solution or setting up environment
        return 0.3
    elif any(kw in a for kw in info_keywords):
        # Exploring the environment
        return 0.1
    else:
        # For unknown or generic actions, assume a small positive probability of progress
        # as long as no error was encountered.
        return 0.05