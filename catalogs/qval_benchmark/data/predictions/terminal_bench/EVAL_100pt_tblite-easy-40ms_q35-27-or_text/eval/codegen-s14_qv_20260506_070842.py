def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    q_value = 0.5
    
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Strong completion indicators - high Q-value boost
    completion_keywords = ['success', 'completed', 'done', 'finished', 'passed',
                           'verified', 'correct', '✓', 'ok', 'true', 'exit code 0',
                           'test passed', 'all tests passed', 'verification passed']
    for keyword in completion_keywords:
        if keyword in next_state_lower:
            q_value = min(q_value + 0.45, 1.0)
            break
    
    # Error indicators - reduce Q-value
    error_keywords = ['error', 'failed', 'fail', 'exception', 'traceback',
                      'permission denied', 'not found', 'no such file', 'wrong',
                      'incorrect', 'invalid', 'exit code 1', 'exit code 2', 'return code 1']
    for keyword in error_keywords:
        if keyword in next_state_lower:
            q_value = max(q_value - 0.35, 0.0)
            break
    
    # Productive command patterns
    productive_commands = ['python', 'python3', 'pip', 'pip3', 'apt', 'yum',
                           'curl', 'wget', 'git', 'ssh', 'scp', 'rsync',
                           'tar', 'zip', 'unzip', 'chmod', 'chown', 'mkdir',
                           'touch', 'cp', 'mv', 'rm', 'grep', 'sed', 'awk',
                           'find', 'cat', 'head', 'tail', 'diff', 'make',
                           'gcc', 'g++', 'npm', 'node', 'java', 'javac',
                           'docker', 'kubectl', 'kubectl', 'systemctl', 'service']
    for cmd in productive_commands:
        if cmd in action_lower:
            q_value = min(q_value + 0.08, 1.0)
            break
    
    # File creation/modification indicators
    file_indicators = ['created', 'wrote', 'saved', 'generated', 'output',
                       'written to', 'file created', 'directory created']
    for keyword in file_indicators:
        if keyword in next_state_lower:
            q_value = min(q_value + 0.12, 1.0)
            break
    
    # Progress indicators
    progress_indicators = ['step', 'phase', 'stage', 'progress', 'loading',
                           'processing', 'building', 'compiling', 'installing',
                           'downloading', 'extracting', 'running', 'executing']
    for keyword in progress_indicators:
        if keyword in next_state_lower:
            q_value = min(q_value + 0.06, 1.0)
            break
    
    # Detect state change (progress signal)
    if next_state != state:
        if len(next_state) > len(state) + 5:
            q_value = min(q_value + 0.05, 1.0)
        elif len(next_state) < len(state) - 5:
            q_value = max(q_value - 0.02, 0.0)
    
    # Penalize empty or minimal actions
    if action.strip() == '' or len(action.strip()) < 3:
        q_value = max(q_value - 0.2, 0.0)
    
    # Penalize repeated states (stuck)
    if next_state == state and len(action.strip()) > 3:
        q_value = max(q_value - 0.15, 0.0)
    
    # Pattern for successful output (numbers, paths, hashes)
    if re.search(r'\b[A-Fa-f0-9]{8,}\b', next_state) or re.search(r'/\w+', next_state):
        q_value = min(q_value + 0.03, 1.0)
    
    # Clamp to valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value