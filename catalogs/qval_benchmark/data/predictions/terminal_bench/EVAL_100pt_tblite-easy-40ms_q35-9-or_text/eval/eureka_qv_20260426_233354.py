def signal_function(state: str, action: str, next_state: str):
    success_score = 0.0
    failure_score = 0.0
    progress_score = 0.0
    efficiency_score = 0.0
    safety_score = 0.0
    terminal_score = 0.0
    
    # Success indicators - task completion
    success_patterns = [
        'task completed', 'task finished', 'all done', 'successfully',
        'verification passed', 'verified', 'passed', 'success', 'done',
        'completed', 'finished', 'solved', 'ok', 'true', 'valid',
        'exit code 0', 'code 0', 'return code 0', 'no errors',
        'no issues', 'clean', 'ready', 'initialized', 'shell ready',
        'output written', 'saved', 'created', 'generated', 'built'
    ]
    
    # Failure indicators - task failure or errors
    failure_patterns = [
        'error', 'fail', 'failed', 'exception', 'syntax error',
        'permission denied', 'denied', 'invalid', 'timeout',
        'killed', 'segfault', 'crash', 'broken pipe', 'refused',
        'command not found', 'no space', 'cannot access', 'interrupted',
        'aborted', 'fatal', 'traceback', 'connection', 'failed',
        'exit code non-zero', 'return code non-zero', 'error code',
        'failed to', 'could not', 'unable to', 'not found',
        'segmentation fault', 'core dump', 'memory error', 'out of memory'
    ]
    
    # Progress indicators - active work happening
    progress_patterns = [
        'processing', 'building', 'compiling', 'training', 'installing',
        'downloading', 'extracting', 'copying', 'moving', 'uploading',
        'fetching', 'parsing', 'analyzing', 'generating', 'creating',
        'writing', 'reading', 'saving', 'epoch', 'iter', 'step',
        'percent', 'remaining', 'loading', 'running', 'executing',
        'pending', 'waiting', 'working', 'active', 'in progress',
        'progress', 'copying files', 'transferring', 'converting',
        'optimizing', 'tuning', 'fitting', 'evaluating'
    ]
    
    # Terminal ready indicators - shell is interactive and waiting
    terminal_ready_patterns = [
        'root@', '$ ', '# ', 'prompt', 'ready', 'waiting', 'idle',
        'bash:', 'sh:', 'zsh:', 'python:', 'terminal:', 'console:',
        'login:', 'user@', 'hostname', 'command ready', 'interactive',
        'last command', 'exit', 'logout', 'session', 'shell'
    ]
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check success in current state
    for pattern in success_patterns:
        if pattern in state_lower:
            success_score += 0.35
            break
    
    # Check failure in current state
    for pattern in failure_patterns:
        if pattern in state_lower:
            failure_score -= 0.55
            break
    
    # Check progress in current state
    for pattern in progress_patterns:
        if pattern in state_lower:
            progress_score += 0.18
            break
    
    # Check terminal ready in current state
    for pattern in terminal_ready_patterns:
        if pattern in state_lower:
            terminal_score += 0.25
            break
    
    # Check next state for success (strong signal)
    for pattern in success_patterns:
        if pattern in next_state_lower:
            success_score += 0.55
            break
    
    # Check next state for failure
    for pattern in failure_patterns:
        if pattern in next_state_lower:
            failure_score -= 0.6
            break
    
    # Check next state for terminal ready (good transition)
    for pattern in terminal_ready_patterns:
        if pattern in next_state_lower:
            terminal_score += 0.28
            break
    
    # Check next state for progress
    for pattern in progress_patterns:
        if pattern in next_state_lower:
            progress_score += 0.12
            break
    
    # Action quality scoring based on command type
    action_length = len(action)
    if 10 <= action_length <= 150:
        efficiency_score += 0.18
    elif 5 <= action_length < 10:
        efficiency_score += 0.12
    elif 0 < action_length < 5:
        efficiency_score += 0.08
    elif action_length > 200:
        efficiency_score -= 0.12
    
    # Action type bonuses - common successful commands
    if any(cmd in action_lower for cmd in ['python', 'python3', 'pip', 'apt', 'git', 'sudo', 'cat', 'grep', 'echo', 'cd', 'mkdir', 'touch', 'chmod', 'chown', 'rm', 'mv', 'cp', 'ls', 'pwd', 'whoami', 'env', 'date', 'clear', 'history', 'exit']):
        efficiency_score += 0.12
    
    # State length as efficiency indicator
    state_length = len(state)
    if state_length < 100:
        efficiency_score += 0.08
    elif state_length < 300:
        efficiency_score += 0.12
    
    # Penalty for very large outputs (might indicate issues or loops)
    if state_length > 800:
        safety_score -= 0.12
    elif state_length > 400:
        safety_score -= 0.06
    
    # Transition efficiency
    state_length_diff = abs(state_length - len(next_state))
    if state_length_diff < 100:
        efficiency_score += 0.08
    elif state_length_diff < 300:
        efficiency_score += 0.04
    
    # Check for error in next state (strong negative signal)
    if any(pattern in next_state_lower for pattern in ['error', 'fail', 'denied', 'invalid', 'exception', 'traceback']):
        failure_score -= 0.45
    
    # Check for success in next state (strong positive signal)
    if any(pattern in next_state_lower for pattern in ['success', 'done', 'passed', 'verified', 'completed', 'exit code 0', 'return code 0']):
        success_score += 0.65
    
    # Additional safety check for repeated error patterns
    if 'error' in next_state_lower and 'error' in state_lower:
        failure_score -= 0.2
    
    # Bonus for clean terminal state (no error markers)
    if 'error' not in next_state_lower and 'fail' not in next_state_lower:
        safety_score += 0.05
    
    # Calculate total Q-value estimate
    total = success_score + failure_score + progress_score + efficiency_score + safety_score + terminal_score
    
    return total, {
        "success_score": success_score,
        "failure_score": failure_score,
        "progress_score": progress_score,
        "efficiency_score": efficiency_score,
        "safety_score": safety_score,
        "terminal_score": terminal_score,
    }