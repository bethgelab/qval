def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check for completion indicators in next_state
    completion_patterns = [
        r'pass', r'success', r'completed', r'finished', 
        r'✓', r'✔', r'test passed', r'verifier',
        r'exit code 0', r'OK', r'all tests passed'
    ]
    
    # Check for failure indicators
    failure_patterns = [
        r'fail', r'error', r'failed', r'incorrect',
        r'test failed', r'permission denied', r'not found', 
        r'No such file', r'command not found', r'exception'
    ]
    
    # Check if next_state indicates success
    success_score = 0.0
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            success_score = 1.0
            break
    
    # Check for failure
    failure_score = 0.0
    for pattern in failure_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            failure_score = 1.0
            break
    
    # Analyze action quality based on common productive terminal commands
    productive_actions = [
        'cd ', 'ls ', 'cat ', 'grep ', 'find ', 'cp ', 'mv ', 
        'rm ', 'mkdir ', 'touch ', 'echo ', 'python', 'bash',
        'sh ', 'make', 'gcc', 'g++', 'pip', 'install', 'chmod',
        'chown', 'ln ', 'tar', 'zip', 'unzip', 'git', 'curl',
        'wget', 'ssh', 'scp', 'rsync', 'sed', 'awk', 'sort',
        'uniq', 'head', 'tail', 'wc', 'diff', 'patch', 'build',
        'compile', 'test', 'run', 'execute', 'create', 'write',
        'read', 'open', 'save', 'submit', 'verify'
    ]
    
    action_score = 0.0
    for cmd in productive_actions:
        if action.lower().strip().startswith(cmd.lower()):
            action_score = 0.6
            break
    
    # Penalize obviously unproductive actions
    unproductive_actions = ['quit', 'exit', 'cancel', 'abort', 'stop']
    for bad_action in unproductive_actions:
        if bad_action in action.lower():
            action_score = 0.0
    
    # Check for progress indicators in next_state
    progress_score = 0.0
    
    # Look for file creation/modification
    if re.search(r'created|modified|updated|written|saved|written to', next_state, re.IGNORECASE):
        progress_score = 0.5
    
    # Look for directory changes
    if re.search(r'changed|moved|entered|switched to', next_state, re.IGNORECASE):
        progress_score = max(progress_score, 0.3)
    
    # Look for task completion markers
    if re.search(r'task.*complete|goal.*achieved|done|done!', next_state, re.IGNORECASE):
        progress_score = 0.7
    
    # Check for partial progress (files exist, commands executed)
    if re.search(r'file.*found|command.*executed|output|result', next_state, re.IGNORECASE):
        progress_score = max(progress_score, 0.2)
    
    # Penalize if state shows we're stuck or looping
    if re.search(r'loop|again|repeated|same|unchanged', next_state, re.IGNORECASE):
        progress_score = max(0.0, progress_score - 0.3)
    
    # Combine scores with weights
    q_value = 0.5 * success_score + 0.2 * action_score + 0.3 * progress_score
    
    # Apply failure penalty
    if failure_score > 0.5:
        q_value = max(0.0, q_value - 0.5)
    
    # Ensure value is in reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)