import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a, s') for a terminal-based task.
    The value represents the estimated probability of success given the current state,
    the action taken, and the resulting next state.
    """
    # Start with a neutral Q-value
    q = 0.5

    # Keywords indicating progress or success
    success_phrases = [
        "successfully", "all tests passed", "completed", "correct", 
        "verified", "done", "success", "task complete"
    ]
    
    # Keywords indicating failure or errors
    failure_phrases = [
        "not found", "permission denied", "syntax error", "failed", 
        "error", "invalid", "incorrect", "exception", "wrong"
    ]

    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()

    # 1. Analyze the next state for success/failure indicators
    # Check failure phrases first to avoid "not found" being caught by "found"
    failure_detected = False
    for phrase in failure_phrases:
        if phrase in next_state_lower:
            q -= 0.3
            failure_detected = True
            break
            
    if not failure_detected:
        for phrase in success_phrases:
            if phrase in next_state_lower:
                q += 0.3
                break

    # 2. Evaluate the action taken
    # Actions that are likely final verification or submission steps
    if any(word in action_lower for word in ["submit", "verify", "check", "test"]):
        if any(phrase in next_state_lower for phrase in success_phrases):
            q += 0.2  # High value for a successful final step
        elif any(phrase in next_state_lower for phrase in failure_phrases):
            q -= 0.2  # Penalty for a failed final step
        else:
            q += 0.05  # Slight value for attempting the final step

    # Productive administrative/editing commands
    productive_cmds = ["vim", "nano", "sed", "grep", "chmod", "chown", "pip install", "apt-get", "mkdir"]
    if any(cmd in action_lower for cmd in productive_cmds):
        q += 0.05

    # Exploratory commands (helpful, but low intrinsic value)
    exploratory_cmds = ["ls", "pwd", "cat", "find", "head", "tail", "echo"]
    if any(cmd in action_lower for cmd in exploratory_cmds):
        q += 0.02

    # 3. Relative improvement analysis
    # Did the number of error indicators decrease?
    state_error_count = sum(1 for phrase in failure_phrases if phrase in state_lower)
    next_error_count = sum(1 for phrase in failure_phrases if phrase in next_state_lower)
    
    if next_error_count < state_error_count:
        q += 0.15  # Improving the situation
    elif next_error_count > state_error_count:
        q -= 0.15  # Making the situation worse

    # 4. Redundancy penalty
    # If the action didn't change the state meaningfully
    if next_state == state:
        q -= 0.1

    # Clip the final Q-value to the range [0.0, 1.0]
    return max(0.0, min(1.0, q))