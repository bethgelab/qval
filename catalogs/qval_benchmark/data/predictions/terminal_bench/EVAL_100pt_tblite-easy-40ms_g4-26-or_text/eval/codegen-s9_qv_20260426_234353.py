import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given terminal state, action, and next state.
    The estimation is based on detecting success, errors, and the productivity of the command.
    """
    ns_lower = next_state.lower()
    
    # 1. Error Indicators
    # We look for specific error patterns that typically indicate a failed command or a dead end.
    # We use more specific patterns (like 'error:' or 'error ') to avoid false positives 
    # in cases like "No error found".
    error_patterns = [
        r'command not found', r'permission denied', r'no such file', 
        r'no such directory', r'syntaxerror', r'traceback', 
        r'segmentation fault', r'failed ', r'error:', r'error ', 
        r'fatal:', r'not found', r'invalid syntax', r'cannot find', 
        r'could not find', r'is not a directory'
    ]
    for pattern in error_patterns:
        if re.search(pattern, ns_lower):
            return 0.0

    # 2. Success Indicators
    # We look for common indicators that the goal or a step in the goal has been achieved.
    success_patterns = [
        r'success', r'passed', r'correct', r'done', r'completed', 
        r'\bok\b', r'finished'
    ]
    for pattern in success_patterns:
        if re.search(pattern, ns_lower):
            return 1.0

    # 3. Action/Command Analysis
    # If no immediate success or error is found, we estimate the value based on the 
    # type of command executed.
    if not action.strip():
        return 0.0
        
    # Extract the primary command (the first word) from the action
    cmd_match = re.match(r'^\s*([^\s]+)', action)
    if not cmd_match:
        return 0.2
    cmd = cmd_match.group(1).lower()

    # Categorize commands into functional groups
    exploratory_cmds = {
        'ls', 'cat', 'grep', 'find', 'pwd', 'cd', 'python', 'python3', 
        'bash', 'git', 'diff', 'head', 'tail', 'wc', 'file', 'which', 
        'whereis', 'stat', 'du', 'df', 'nm', 'objdump', 'strings'
    }
    modifying_cmds = {
        'mv', 'cp', 'rm', 'mkdir', 'chmod', 'chown', 'sed', 'echo', 
        'touch', 'apt', 'pip', 'wget', 'curl', 'nano', 'vi', 'vim', 
        'apt-get', 'pip3'
    }
    useless_cmds = {'clear', 'exit', 'history', 'help'}

    if cmd in useless_cmds:
        return 0.1
    if cmd in exploratory_cmds:
        # Exploratory commands are highly valued as they provide context needed to progress
        return 0.6
    if cmd in modifying_cmds:
        # Modifying commands are valued as they are necessary to reach many terminal goals
        return 0.4
        
    # Default value for unknown or generic commands that do not produce error messages
    return 0.3