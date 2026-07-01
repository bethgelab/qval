def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    # Base Q-value starts neutral
    q_value = 0.0
    
    # Success indicators in next_state
    success_patterns = [
        r'success', r'completed', r'passed', r'done', r'finished',
        r'exit code 0', r'0 errors', r'✓', r'✔', r'OK',
        r'created', r'generated', r'written', r'saved', r'wrote',
        r'test passed', r'all tests', r'verified'
    ]
    
    # Error indicators
    error_patterns = [
        r'error', r'failed', r'exception', r'fail',
        r'cannot', r'permission denied', r'not found',
        r'invalid', r'incorrect', r'missing', r'no such',
        r'traceback', r'assertion'
    ]
    
    # Check for success indicators
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.12
    
    # Check for error indicators (penalize)
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value -= 0.10
    
    # Check if action appears to be making progress (valid commands)
    action_progress_indicators = [
        r'cd\s+', r'ls\s+', r'cat\s+', r'echo\s+', r'cp\s+', r'mv\s+',
        r'python', r'bash', r'sh\s+', r'\./', r'chmod', r'grep',
        r'sed', r'awk', r'find', r'tar', r'zip', r'unzip',
        r'mkdir', r'rm\s+', r'rm -rf', r'chmod', r'chown',
        r'export', r'source', r'pip', r'install', r'import',
        r'openssl', r'gpg', r'hash', r'encrypt', r'decrypt',
        r'curl', r'wget', r'git', r'commit', r'push', r'pull'
    ]
    
    for pattern in action_progress_indicators:
        if re.search(pattern, action, re.IGNORECASE):
            q_value += 0.03
            break  # Only count once for valid action
    
    # Check for file/directory creation indicators in next_state
    file_indicators = [
        r'\.txt', r'\.py', r'\.json', r'\.csv', r'\.log',
        r'\.sh', r'\.bash', r'\.md', r'\.yaml', r'\.yml',
        r'\.json', r'\.xml', r'\.html', r'\.css', r'\.js',
        r'\.tar', r'\.gz', r'\.zip', r'\.db', r'\.sql'
    ]
    
    for pattern in file_indicators:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.08
            break
    
    # Check for task-specific completion patterns
    task_completion = [
        r'answer', r'result', r'output', r'final', r'flag',
        r'secret', r'key', r'token', r'password', r'credential',
        r'decoded', r'encrypted', r'hashed', r'signed',
        r'model', r'trained', r'predict', r'classify'
    ]
    
    for pattern in task_completion:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.10
            break
    
    # Check if next_state differs meaningfully from state (progress)
    if len(next_state.strip()) > len(state.strip()) * 0.5:
        q_value += 0.05
    
    # Penalize if action seems destructive without recovery
    destructive_actions = [r'rm\s+-rf', r'drop\s+table', r'delete\s+from']
    for pattern in destructive_actions:
        if re.search(pattern, action, re.IGNORECASE):
            if not re.search(r'confirmed|yes|proceed', next_state, re.IGNORECASE):
                q_value -= 0.15
                break
    
    # Bonus for reaching near-completion patterns
    near_complete = [
        r'final\s+answer', r'submit', r'answer:', r'flag:',
        r'your\s+flag', r'the\s+flag', r'congratulations'
    ]
    
    for pattern in near_complete:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.20
            break
    
    # Ensure value stays in reasonable range [-1.0, 1.0]
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value