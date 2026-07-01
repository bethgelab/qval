def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value starts at neutral
    q_value = 0.0
    
    # Check for success indicators in next_state
    success_patterns = [
        r'passed',
        r'success',
        r'completed',
        r'test.*ok',
        r'all tests',
        r'✓',
        r'PASS',
        r'OK',
        r'verified',
    ]
    
    success_count = 0
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            success_count += 1
    
    # Check for error indicators
    error_patterns = [
        r'error',
        r'fail',
        r'exception',
        r'traceback',
        r'permission denied',
        r'not found',
        r'no such',
        r'invalid',
        r'failed',
        r'cannot',
        r'unable',
    ]
    
    error_count = 0
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            error_count += 1
    
    # Calculate progress based on success vs error patterns
    if success_count > 0:
        q_value += 0.2 * success_count
    
    if error_count > 0:
        q_value -= 0.15 * error_count
    
    # Check if action was productive (created files, ran commands)
    productive_patterns = [
        r'created',
        r'wrote',
        r'copied',
        r'moved',
        r'installed',
        r'updated',
        r'added',
        r'generated',
        r'written',
        r'saved',
    ]
    
    for pattern in productive_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.1
    
    # Penalize if action was likely wrong (no output or error)
    if len(next_state.strip()) < 10 and error_count == 0:
        q_value -= 0.1
    
    # Check for specific task completion markers
    completion_markers = [
        r'verifier',
        r'test.*passed',
        r'all tests passed',
        r'submit',
        r'done',
        r'task complete',
        r'finished',
    ]
    
    for marker in completion_markers:
        if re.search(marker, next_state, re.IGNORECASE):
            q_value += 0.5
    
    # Check if state changed significantly (progress indicator)
    if len(next_state) > len(state) + 20:
        q_value += 0.05
    
    # Check for file operations in action
    file_ops = [r'\.py$', r'\.sh$', r'\.txt$', r'\.json$', r'\.csv$', r'\.md$']
    for pattern in file_ops:
        if re.search(pattern, action):
            q_value += 0.02
    
    # Clamp to reasonable range
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value