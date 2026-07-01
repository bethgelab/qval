def signal_function(state: str, action: str, next_state: str) -> float:
    # Analyze terminal state for Q-value estimation
    # Higher values indicate more favorable state-action pairs
    
    score = 0.0
    
    # Strong negative signals - errors that likely indicate task failure
    error_patterns = ['error', 'failed', 'cannot', 'denied', 'permission denied', 
                      'no such file', 'not found', 'does not exist', 'abort', 
                      'fatal', 'exception', 'traceback', 'syntax error', 'invalid']
    next_lower = next_state.lower()
    for pattern in error_patterns:
        if pattern in next_lower:
            score -= 0.8
            break
    
    # Strong positive signals - success indicators
    success_patterns = ['success', 'completed', 'done', 'pass', 'verified', 
                        'ok', 'true', 'accepted', 'valid', 'correct']
    for pattern in success_patterns:
        if pattern in next_lower:
            score += 0.7
            break
    
    # Progress indicators - task advancement signals
    progress_patterns = ['created', 'wrote', 'saved', 'generated', 'built', 
                         'compiled', 'installed', 'copied', 'moved', 'deleted',
                         'updated', 'changed', 'modified', 'added', 'removed']
    for pattern in progress_patterns:
        if pattern in next_lower:
            score += 0.4
            break
    
    # Command execution quality
    action_lower = action.lower()
    
    # Reward productive commands
    productive_commands = ['cat', 'ls', 'grep', 'find', 'cp', 'mv', 'rm', 
                           'mkdir', 'touch', 'echo', 'python', 'pip', 'curl',
                           'wget', 'chmod', 'chown', 'ssh', 'scp', 'git',
                           'make', 'gcc', 'clang', 'npm', 'yarn', 'docker']
    for cmd in productive_commands:
        if action_lower.startswith(cmd):
            score += 0.2
            break
    
    # Penalize potentially destructive commands without clear purpose
    dangerous_commands = ['rm -rf', 'rm -r', 'dd if=', '> /dev', 'chmod 000']
    for cmd in dangerous_commands:
        if cmd in action_lower:
            score -= 0.3
            break
    
    # Output quality - non-empty output suggests activity
    if len(next_state.strip()) > 0:
        # More output often indicates more information/progress
        output_length = len(next_state)
        if output_length > 100:
            score += 0.15
        elif output_length > 10:
            score += 0.1
    
    # Check if state appears to show a clean shell prompt (good continuation)
    prompt_patterns = ['$', '#', 'user@', '(venv)', '(conda)']
    has_prompt = any(p in next_state for p in prompt_patterns)
    if has_prompt and not any(e in next_lower for e in ['error', 'failed']):
        score += 0.1
    
    # State stability - check if we're making forward progress
    if len(state) > 0 and len(next_state) > len(state):
        score += 0.05  # Information gain
    
    # Clamp to reasonable Q-value range [-1, 1]
    return max(-1.0, min(1.0, score))