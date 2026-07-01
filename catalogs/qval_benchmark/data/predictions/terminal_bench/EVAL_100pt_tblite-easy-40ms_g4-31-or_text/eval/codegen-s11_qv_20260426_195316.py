import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) based on the transition from state to next_state
    given the action taken in a terminal-based environment.
    """
    # Base estimate: neutral starting point
    q_value = 0.3

    # 1. Failure Detection
    # Identify common shell error messages and failure indicators.
    failure_patterns = [
        r"command not found",
        r"Permission denied",
        r"No such file or directory",
        r"Syntax error",
        r"failed",
        r"error:",
        r"Invalid",
        r"cannot open",
        r"fatal error"
    ]
    
    is_failure = False
    for pat in failure_patterns:
        if re.search(pat, next_state, re.IGNORECASE):
            is_failure = True
            break
    
    if is_failure:
        q_value -= 0.4

    # 2. Success/Progress Detection
    # Identify markers that indicate a command achieved its intended purpose.
    success_patterns = [
        r"successfully",
        r"completed",
        r"done",
        r"fixed",
        r"passed",
        r"updated",
        r"created",
        r"written"
    ]
    
    found_success_marker = False
    for pat in success_patterns:
        if re.search(pat, next_state, re.IGNORECASE):
            found_success_marker = True
            break
    
    if found_success_marker:
        q_value += 0.3

    # 3. Action Analysis
    # Categorize the action to determine its potential impact on the goal.
    
    # Exploration/Information gathering
    if any(cmd in action for cmd in ["ls", "pwd", "cat", "grep", "find", "head", "tail", "stat"]):
        q_value += 0.1
    
    # Configuration/Modification
    if any(cmd in action for cmd in ["sed", "echo", "vi", "nano", "cp", "mv", "rm", "chmod", "chown", "mkdir"]):
        q_value += 0.2
        
    # Execution/Processing
    if any(cmd in action for cmd in ["python", "bash", "sh", "gcc", "make", "./", "pip"]):
        q_value += 0.2
        
    # Finality/Verification (High weight if successful)
    if any(cmd in action for cmd in ["submit", "verify", "pytest", "test", "check"]):
        if not is_failure:
            # If the verification action succeeded, we are likely at the goal
            q_value += 0.5
        else:
            # Failed verification is a setback
            q_value -= 0.2

    # 4. State Transition Analysis
    # If the next state provides more useful information (longer output) and didn't error,
    # it's often a sign of productive interaction.
    if not is_failure and len(next_state) > len(state):
        q_value += 0.1

    # 5. Final Constraint
    # Clip the Q-value to the valid range [0.0, 1.0].
    return max(0.0, min(1.0, q_value))