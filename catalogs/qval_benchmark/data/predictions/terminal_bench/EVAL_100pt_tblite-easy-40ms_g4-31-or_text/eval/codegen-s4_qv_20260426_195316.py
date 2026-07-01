import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) based on the transition from state to next_state
    given action a. The Q-value reflects the estimated probability of eventually
    reaching a successful terminal state.
    """
    # Baseline probability of success
    q = 0.5
    
    # Normalize strings for case-insensitive matching
    ns_lower = next_state.lower()
    a_lower = action.lower()
    
    # Define markers for success and failure in terminal output
    success_indicators = [
        "successfully", "success", "passed", "correct", "done", 
        "completed", "verified", "all tests passed", "goal reached"
    ]
    failure_indicators = [
        "error", "failed", "not found", "denied", "invalid", 
        "syntax error", "cannot open", "no such file", "permission denied",
        "wrong answer", "incorrect"
    ]
    
    # Check for indicators in the resulting state
    has_success = any(ind in ns_lower for ind in success_indicators)
    has_failure = any(ind in ns_lower for ind in failure_indicators)
    
    # Initial adjustments based on the output of the action
    if has_success and not has_failure:
        q = 0.8
    elif has_failure and not has_success:
        q = 0.2
    elif has_success and has_failure:
        q = 0.4  # Mixed signals
        
    # Higher weights for actions intended to finalize or verify the task
    # If a verification/submission action leads to a success message, Q is very high.
    is_submission = any(cmd in a_lower for cmd in ["submit", "verify", "check", "test"])
    if is_submission:
        if has_success and not has_failure:
            q = 0.95
        elif has_failure:
            q = 0.05
    
    # Heuristic: actions that modify the system/files are generally positive 
    # if they do not result in an explicit error.
    is_modification = any(cmd in a_lower for cmd in [
        "echo", "vim", "nano", "sed", "mv", "cp", "touch", 
        "python", "gcc", "openssl", "make", "pip", "apt"
    ])
    if is_modification and not has_failure:
        q = max(q, 0.6)
        
    # Heuristic: exploratory commands (listing files, etc.) are slightly positive 
    # if they don't fail, as they help the agent gain information.
    is_exploration = any(cmd in a_lower for cmd in ["ls", "pwd", "cat", "grep", "find", "whoami", "df", "du"])
    if is_exploration and not has_failure:
        q = max(q, 0.55)

    # Penalty if the action caused no visible change in the terminal state
    if next_state == state:
        q -= 0.1
        
    # Ensure the return value is clamped between 0.0 and 1.0
    return float(max(0.0, min(1.0, q)))