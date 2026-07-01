def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.5  # Baseline estimate
    
    # Success indicators - strong positive signals
    success_patterns = ['success', 'completed', 'done', 'passed', 'ok', '✓', 'exit code 0', 'return code 0', 'all tests passed', 'verification successful']
    for pattern in success_patterns:
        if pattern.lower() in next_state.lower():
            q_value += 0.2
    
    # Error indicators - strong negative signals
    error_patterns = ['error', 'failed', 'exception', 'traceback', 'exit code 1', 'exit code 2', '✗', 'permission denied', 'not found', 'no such file', 'syntax error', 'command not found']
    for pattern in error_patterns:
        if pattern.lower() in next_state.lower():
            q_value -= 0.25
    
    # Progress indicators - moderate positive signals
    progress_patterns = ['created', 'wrote', 'generated', 'saved', 'downloaded', 'extracted', 'copied', 'built', 'compiled', 'installed']
    for pattern in progress_patterns:
        if pattern.lower() in next_state.lower():
            q_value += 0.1
    
    # Completion-related actions
    completion_actions = ['submit', 'verify', 'test', 'check', 'validate', 'finish', 'end']
    for action_type in completion_actions:
        if action_type.lower() in action.lower():
            q_value += 0.15
    
    # Early stage indicators - negative signal (far from goal)
    early_patterns = ['initializing', 'starting', 'setup', 'beginning', 'preparing', 'loading']
    for pattern in early_patterns:
        if pattern.lower() in next_state.lower():
            q_value -= 0.1
    
    # File creation success indicators
    file_patterns = ['file created', 'directory created', 'output written', 'result saved']
    for pattern in file_patterns:
        if pattern.lower() in next_state.lower():
            q_value += 0.15
    
    # Command execution success
    exec_patterns = ['command executed', 'process completed', 'task finished', 'job done']
    for pattern in exec_patterns:
        if pattern.lower() in next_state.lower():
            q_value += 0.1
    
    # Penalize if action is empty or just whitespace
    if not action or action.strip() == '':
        q_value -= 0.3
    
    # Clamp to [0, 1] range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value