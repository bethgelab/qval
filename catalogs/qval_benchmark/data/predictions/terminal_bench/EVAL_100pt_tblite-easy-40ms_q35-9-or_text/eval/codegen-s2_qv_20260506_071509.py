def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    success_indicators = [
        r'✓', r'success', r'ok', r'completed', r'created', r'wrote',
        r'exit', r'0 errors', r'no errors', r'PASS', r'passed', r'root@',
        r'$', r'$', r'~$', r'~', r'git commit', r'git push', r'git push',
        r'python', r'exit', r'0', r'none', r'empty', r'no issues'
    ]
    
    error_indicators = [
        r'error', r'fail', r'Error', r'FAIL', r'failed', r'exception',
        r'traceback', r'invalid', r'permission denied', r'not found',
        r'syntax error', r'command not found', r'no such file', r'broken',
        r'crash', r'aborted', r'killed', r'timeout', r'connection refused'
    ]
    
    success_count = sum(1 for pattern in success_indicators if re.search(pattern, next_state, re.IGNORECASE))
    error_count = sum(1 for pattern in error_indicators if re.search(pattern, next_state, re.IGNORECASE))
    
    base_signal = success_count - error_count
    
    state_len = len(state)
    next_len = len(next_state)
    
    if state_len == 0:
        length_bonus = 0.0
    elif next_len <= state_len:
        length_bonus = 0.3
    elif next_len <= state_len + 100:
        length_bonus = 0.2
    elif next_len <= state_len + 300:
        length_bonus = 0.1
    else:
        length_bonus = 0.0
    
    action_len = len(action)
    if action_len == 0 or action.strip() == '':
        action_bonus = 0.0
    elif action_len <= 50:
        action_bonus = 0.2
    elif action_len <= 100:
        action_bonus = 0.1
    else:
        action_bonus = 0.0
    
    q_value = base_signal * 0.5 + length_bonus * 0.3 + action_bonus * 0.2
    
    return max(0.0, min(1.0, q_value))