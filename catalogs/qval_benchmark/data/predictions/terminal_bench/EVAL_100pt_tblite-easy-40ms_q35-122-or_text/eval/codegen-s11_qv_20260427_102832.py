def signal_function(state: str, action: str, next_state: str) -> float:
    """Estimate Q-value based on state, action, and next_state analysis for terminal tasks."""
    
    # Base estimate - neutral starting point
    q_value = 0.5
    
    # Check for explicit success in next_state (terminal condition met)
    success_keywords = ['success', 'pass', 'verified', 'completed', 'done', 'ok', 'correct', 'passed', 'true', 'exit code 0']
    for keyword in success_keywords:
        if keyword in next_state.lower():
            q_value = 1.0
            return q_value
    
    # Check for explicit failure in next_state (terminal condition failed)
    failure_keywords = ['error', 'fail', 'exception', 'traceback', 'not found', 'permission denied', 'denied', 'invalid', 'exit code 1', 'exit code 127']
    for keyword in failure_keywords:
        if keyword in next_state.lower():
            q_value = 0.0
            return q_value
    
    # Analyze action quality - constructive vs destructive
    action_lower = action.lower()
    
    # Constructive commands that typically lead to task completion
    constructive = ['python', 'bash', 'sh', 'make', 'build', 'compile', 'test', 'run', 'check', 'verify', 
                   'solve', 'create', 'write', 'edit', 'cat', 'echo', 'grep', 'find', 'ls', 'cd', 
                   'mkdir', 'touch', 'chmod', 'cp', 'mv', 'diff', 'patch', 'git', 'install', 'pip']
    
    # Destructive or non-productive commands
    destructive = ['rm -rf', 'rm -f', 'clear', 'reset', 'exit', 'quit', 'logout']
    
    action_bonus = 0.0
    for cmd in constructive:
        if cmd in action_lower:
            action_bonus += 0.05
    
    for cmd in destructive:
        if cmd in action_lower:
            action_bonus -= 0.1
    
    # Check for progress indicators in next_state
    progress_keywords = ['created', 'written', 'saved', 'generated', 'output', 'result', 'found', 'located', 'updated', 'modified']
    progress_bonus = 0.0
    for keyword in progress_keywords:
        if keyword in next_state.lower():
            progress_bonus += 0.05
    
    # Check for file/directory creation (indicates progress)
    if 'total' in next_state.lower() or 'bytes' in next_state.lower():
        progress_bonus += 0.03
    
    # Penalize empty or unchanged states (wasted step)
    if len(next_state.strip()) < 10 and len(state.strip()) > 10:
        q_value -= 0.1
    
    # Calculate final Q-value
    q_value = q_value + action_bonus + progress_bonus
    
    # Clamp to valid probability range [0.0, 1.0]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value