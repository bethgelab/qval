import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value of taking an action in a terminal-based environment.
    Uses keyword detection and heuristic analysis of the terminal output.
    """
    ns = next_state.lower()
    a = action.lower().strip()
    
    # 1. Detect immediate failure indicators
    # Most terminal errors are clear indicators of a bad step.
    error_patterns = [
        r'\berror\b', 
        r'failed', 
        r'not found', 
        r'no such file', 
        r'no such directory', 
        r'permission denied', 
        r'syntaxerror', 
        r'traceback', 
        r'exception', 
        r'invalid', 
        r'denied'
    ]
    for pattern in error_patterns:
        if re.search(pattern, ns):
            return 0.0
    
    # 2. Detect immediate success indicators
    # If the terminal output signals success, the Q-value is maximized.
    success_patterns = [
        r'success', 
        r'passed', 
        r'correct', 
        r'done', 
        r'completed', 
        r'all tests passed', 
        r'result: 1', 
        r'all passed'
    ]
    for pattern in success_patterns:
        if re.search(pattern, ns):
            return 1.0
            
    # 3. Heuristic for productive vs. unproductive steps
    # A step is productive if it's an investigative command and generates output.
    productive_cmds = [
        'ls', 'cat', 'cd', 'pwd', 'grep', 'find', 'python', 'pip', 
        'git', 'mkdir', 'cp', 'mv', 'rm', 'diff', 'head', 'tail', 
        'nano', 'vi', 'vim', 'bash', 'sh', 'python3', 'pip3', 'pytest'
    ]
    
    is_productive_cmd = any(a.startswith(cmd) for cmd in productive_cmds)
    
    # A terminal state is "just a prompt" if it's a single short line ending in $ or #.
    # If the output is not just a prompt, it means the command actually produced data.
    lines = next_state.strip().split('\n')
    is_not_just_prompt = True
    if len(lines) <= 1:
        trimmed = lines[0].strip() if lines else ""
        if (trimmed.endswith('$') or trimmed.endswith('#')) and len(trimmed) < 35:
            is_not_just_prompt = False
    
    # Assign value based on productivity and output content
    if is_productive_cmd:
        # Command ran and produced output (good) or just returned to prompt (neutral)
        return 0.7 if is_not_just_prompt else 0.5
    elif is_not_just_prompt:
        # An unknown command produced output, suggesting some activity occurred
        return 0.6
    else:
        # Command produced no output or just a prompt, suggesting no progress
        return 0.4