def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value starts at neutral
    q_value = 0.0
    
    # Convert to lowercase for case-insensitive pattern matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Check for success/completion indicators in next_state
    success_patterns = [
        'success', 'done', 'complete', 'pass', 'passed', 'verified',
        'correct', 'answer', 'result', 'output', 'solution', 'task',
        'submit', 'final', 'terminal', 'bench', 'check', 'test'
    ]
    for pattern in success_patterns:
        if pattern in next_state_lower:
            q_value += 0.12
            break
    
    # Check for error/failure indicators (negative signal)
    error_patterns = [
        'error', 'fail', 'failed', 'denied', 'permission', 'not found',
        'missing', 'invalid', 'cannot', 'unable', 'exception', 'traceback'
    ]
    for pattern in error_patterns:
        if pattern in next_state_lower:
            q_value -= 0.25
            break
    
    # Check if action was productive (constructive commands)
    productive_patterns = [
        'cat', 'echo', 'touch', 'mkdir', 'cp', 'mv', 'chmod', 'grep',
        'find', 'python', 'bash', 'sh', 'compile', 'make', 'test',
        'run', 'verify', 'check', 'ls', 'cd', 'pwd', 'head', 'tail',
        'wc', 'sort', 'uniq', 'awk', 'sed', 'install', 'pip', 'apt'
    ]
    for pattern in productive_patterns:
        if pattern in action_lower:
            q_value += 0.08
            break
    
    # Check for destructive actions (penalize slightly)
    destructive_patterns = ['rm ', 'rm -rf', 'delete', 'clear ', 'reset']
    for pattern in destructive_patterns:
        if pattern in action_lower:
            q_value -= 0.1
            break
    
    # Check if next_state shows file/directory creation (progress indicator)
    progress_patterns = [
        'created', 'modified', 'updated', 'written', 'saved', 'file',
        'directory', 'directory created', 'file created', 'output to'
    ]
    for pattern in progress_patterns:
        if pattern in next_state_lower:
            q_value += 0.1
            break
    
    # Check if state shows readiness (good positioning)
    readiness_patterns = [
        'ready', 'available', 'exists', 'found', 'located', 'present',
        'installed', 'configured', 'setup'
    ]
    for pattern in readiness_patterns:
        if pattern in state_lower:
            q_value += 0.05
            break
    
    # Check if action appears to be a verification/confirmation step
    if any(verify in action_lower for verify in ['verify', 'check', 'test', 'submit', 'confirm']):
        q_value += 0.15
    
    # Check if next_state contains evidence of task completion
    completion_indicators = ['100%', 'all done', 'finished', 'completed', '✓', '✔', 'ok']
    for indicator in completion_indicators:
        if indicator in next_state_lower:
            q_value += 0.2
            break
    
    # Penalize empty or minimal next_state (suggests no progress)
    if len(next_state_lower.strip()) < 10 and 'error' not in next_state_lower:
        q_value -= 0.05
    
    # Clamp the value to a reasonable Q-value range
    q_value = max(-1.0, min(1.0, q_value))
    
    return float(q_value)