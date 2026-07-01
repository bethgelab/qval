import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The Q-value represents the estimated probability that the current trajectory will 
    eventually result in a successful task completion (reward 1.0).
    """
    ns = next_state.lower()
    act = action.lower()
    
    # 1. Immediate Success Detection
    # If the next state contains indicators that the verification test passed.
    success_patterns = [
        r'passed', 
        r'success', 
        r'ok\b', 
        r'0 errors?', 
        r'0 failed', 
        r'all tests passed',
        r'test\s+passed',
        r'complete'
    ]
    if any(re.search(p, ns) for p in success_patterns):
        return 1.0
        
    # 2. Immediate/Fatal Failure Detection
    # If the next state contains indicators of catastrophic or terminal errors.
    fatal_errors = [
        r'command not found', 
        r'no such file or directory', 
        r'permission denied',
        r'fatal error',
        r'segmentation fault',
        r'address boundary error'
    ]
    if any(re.search(p, ns) for p in fatal_errors):
        return 0.0
        
    # 3. Heuristic Progress Estimation
    # Identify the category of the command performed.
    is_info_cmd = any(cmd in act for cmd in ['ls', 'cat', 'grep', 'find', 'pwd', 'diff', 'echo', 'head', 'tail', 'stat', 'print'])
    is_edit_cmd = any(cmd in act for cmd in ['mkdir', 'touch', 'cp', 'mv', 'rm', 'chmod', 'chown', 'nano', 'vi', 'vim', 'sed', 'awk', 'echo'])
    is_exec_cmd = any(cmd in act for cmd in ['python', 'bash', 'make', 'gcc', 'g++', 'pip', 'pytest', 'run', 'script', 'node', 'perl'])

    # Check for general errors or failures that aren't necessarily fatal.
    has_error = "error" in ns or "failed" in ns or "exception" in ns
    
    # Baseline probability for a state in the middle of a task.
    score = 0.2 

    if is_exec_cmd:
        # If an execution command was run, check if it produced a non-error result.
        if has_error:
            score = 0.1
        else:
            # If it ran without immediate errors, it's a promising step.
            score = 0.5
    elif is_info_cmd:
        # Information gathering commands are valuable if they produce new output.
        # We use length change as a proxy for information gain.
        if len(next_state) > len(state):
            score = 0.4
        else:
            score = 0.2
    elif is_edit_cmd:
        # File manipulation commands are promising if they don't trigger errors.
        if not has_error:
            score = 0.4
        else:
            score = 0.1
    
    # If we encounter an error but didn't hit the 'is_exec_cmd' logic, 
    # or if the error was minor, apply a small penalty.
    if has_error and score > 0.2:
        score -= 0.1

    # Ensure the result is within [0.0, 1.0].
    return max(0.0, min(1.0, float(score)))