import re

def signal_function(state: str, action: str, next_state: str) -> float:
    score = 0.5
    
    success_indicators = [
        r'success', r'done', r'complete', r'passed', r'ok', r'OK', r'root@', r'$', r'# '
    ]
    
    failure_indicators = [
        r'error', r'fail', r'failed', r'failed to', r'cannot', r'permission denied',
        r'syntax error', r'command not found', r'exit code', r'no such file',
        r'broken pipe', r'connection refused', r'timed out', r'aborted'
    ]
    
    warning_indicators = [
        r'warning', r'warning:', r'warning:', r'note:', r'NOTE'
    ]
    
    for pattern in success_indicators:
        if re.search(pattern, next_state, re.IGNORECASE):
            score += 0.1
    
    for pattern in failure_indicators:
        if re.search(pattern, next_state, re.IGNORECASE):
            score -= 0.15
    
    for pattern in warning_indicators:
        if re.search(pattern, next_state, re.IGNORECASE):
            score -= 0.05
    
    return max(0.0, min(1.0, score))