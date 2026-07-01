import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The Q-value represents the likelihood of successfully completing the task.
    """
    # Initial estimate (neutral)
    q = 0.5
    
    # Normalize inputs for easier pattern matching
    s_low = state.lower()
    a_low = action.lower().strip()
    ns_low = next_state.lower()
    
    # 1. Detect Errors (Significant Penalty)
    # Errors in shell or programming languages indicate the current path is failing.
    error_patterns = [
        "command not found", "no such file", "permission denied",
        "syntax error", "error:", "traceback", "not found",
        "failed to", "invalid syntax", "segmentation fault",
        "nameerror", "typeerror", "filenotfounderror", "moduleerror",
        "error occurred", "fatal error"
    ]
    
    has_error = False
    for pattern in error_patterns:
        if pattern in ns_low:
            has_error = True
            break
    
    if has_error:
        q -= 0.4
        
    # 2. Detect Success/Progress (Significant Boost)
    # Success indicators suggest the task is nearing completion.
    success_patterns = [
        "passed", "success", "done", "completed", "correct",
        "all tests passed", "exit code 0", "successfully", "verified"
    ]
    
    has_success = False
    for pattern in success_patterns:
        if pattern in ns_low:
            has_success = True
            break
            
    if has_success:
        q += 0.5
        
    # 3. Redundancy and Stagnation (Penalty)
    # If the state has not changed significantly, the action was likely a waste of a step.
    if ns_low.strip() == s_low.strip():
        q -= 0.3
    elif len(ns_low) <= len(s_low) + 5:
        # Small changes (like just a prompt update) might indicate no meaningful output.
        # However, we only penalize this if no success was detected.
        if not has_success:
            q -= 0.1

    # 4. Exploration and Information Gain (Small Boost)
    # Inspection commands provide context to solve the task.
    discovery_cmds = {'ls', 'cat', 'grep', 'find', 'pwd', 'head', 'tail', 'diff', 'stat', 'wc', 'which', 'dir'}
    cmd_parts = a_low.split()
    cmd = cmd_parts[0] if cmd_parts else ""
    
    if cmd in discovery_cmds:
        # If the command produced a meaningful change in the state length, it provided info.
        if len(ns_low) > len(s_low) + 10:
            q += 0.1
            
    # 5. Task Execution (Small Boost)
    # Execution commands (e.g., running a script) move the state toward completion.
    execution_cmds = {'python', 'pytest', 'gcc', 'make', 'bash', 'sh', 'submit', 'check', 'run', 'nm', 'ld'}
    if cmd in execution_cmds:
        # A non-error execution is generally a positive step.
        if not has_error:
            q += 0.1

    # 6. Final Clamping
    # Q-values must remain within the [0.0, 1.0] range.
    if q > 1.0:
        q = 1.0
    elif q < 0.0:
        q = 0.0
        
    return float(q)