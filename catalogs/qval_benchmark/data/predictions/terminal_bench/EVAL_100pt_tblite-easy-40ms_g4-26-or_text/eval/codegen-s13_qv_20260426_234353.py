def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given terminal action by analyzing the 
    command's success, error indicators, and informational gain.
    """
    # Pre-process strings for case-insensitive matching
    next_state_l = next_state.lower()
    action_l = action.lower().strip()
    state_l = state.lower()

    # 1. Immediate Failure Detection
    # If the command execution results in common shell errors, the Q-value is low.
    failures = (
        "command not found", 
        "permission denied", 
        "no such file",
        "not a directory", 
        "syntax error", 
        "invalid option",
        "segmentation fault", 
        "traceback", 
        "does not exist", 
        "error:"
    )
    if any(f in next_state_l for f in failures):
        return 0.0

    # 2. Immediate Success Detection
    # If the command results in indicators of task completion, the Q-value is high.
    successes = ("success", "passed", "correct", "flag{", "done", "ok")
    if any(s in next_state_l for s in successes):
        return 1.0

    # 3. Heuristic Estimation for Intermediate Steps
    # We use a baseline and adjust it based on the perceived utility of the action.
    q = 0.3
    
    # Information Gain:
    # We estimate the size of the command output. In terminal-based states, 
    # the next_state usually appends the action and its output to the previous state.
    output_len = len(next_state) - len(state) - len(action)
    
    if output_len > 20:
        q += 0.4  # Substantial output suggests information was retrieved
    elif output_len > 0:
        q += 0.1  # Small output suggests some feedback was received

    # Action Utility:
    # Distinguish between exploratory actions (gathering info) and 
    # transformative actions (changing the environment).
    exploratory = ('ls', 'cat', 'pwd', 'whoami', 'env', 'find', 'grep', 'dir', 'stat', 'echo')
    transformative = ('mkdir', 'touch', 'cp', 'mv', 'rm', 'chmod', 'chown', 'apt', 'pip', 'gcc', 'python', 'make', 'git', 'cd')
    
    if action_l.startswith(exploratory):
        q += 0.1
    elif action_l.startswith(transformative):
        q += 0.1
            
    # Stagnation Penalty:
    # If the action appears to be a repeat of a command already in the history, 
    # it likely provides no new progress.
    if action_l and action_l in state_l:
        q -= 0.2

    # Return the clamped Q-value
    return max(0.0, min(1.0, q))