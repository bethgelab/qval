def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    q_value = 0.5
    
    # Check for completion indicators in next_state
    completion_patterns = [
        r'task.*complete',
        r'success',
        r'passed',
        r'verified',
        r'finished',
        r'done',
        r'✓',
        r'✔',
        r'PASS',
        r'correct',
        r'answer',
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value = 1.0
            return q_value
    
    # Penalize error patterns in next_state
    error_patterns = [
        r'\berror\b',
        r'\bfail\b',
        r'\bexception\b',
        r'\binvalid\b',
        r'not found',
        r'permission denied',
        r'command not found',
        r'no such file',
        r'failed',
        r'\bdenied\b',
    ]
    
    error_count = 0
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            error_count += 1
    
    q_value -= 0.15 * min(error_count, 3)
    
    # Check if action looks like a valid shell command
    valid_command_patterns = [
        r'^cd\s',
        r'^ls\s',
        r'^cat\s',
        r'^echo\s',
        r'^grep\s',
        r'^find\s',
        r'^mkdir\s',
        r'^touch\s',
        r'^rm\s',
        r'^cp\s',
        r'^mv\s',
        r'^python',
        r'^pip',
        r'^git\s',
        r'^chmod\s',
        r'^head\s',
        r'^tail\s',
        r'^wc\s',
        r'^sort\s',
        r'^uniq\s',
        r'^awk\s',
        r'^sed\s',
        r'^diff\s',
        r'^sha256sum',
        r'^md5sum',
        r'^openssl',
        r'^base64',
        r'^curl\s',
        r'^wget\s',
        r'^ssh\s',
        r'^scp\s',
        r'^tar\s',
        r'^zip\s',
        r'^unzip\s',
        r'^echo\s+.*>',
        r'^echo\s+.*>>',
        r'^\.\s+',
        r'^source\s+',
    ]
    
    action_stripped = action.strip()
    action_valid = False
    for pattern in valid_command_patterns:
        if re.search(pattern, action_stripped, re.IGNORECASE):
            action_valid = True
            break
    
    if action_valid:
        q_value += 0.1
    elif action_stripped and len(action_stripped) > 3:
        q_value -= 0.05
    
    # Measure state change as progress indicator
    state_change = abs(len(next_state) - len(state))
    if state_change > 50:
        q_value += 0.05
    elif state_change > 10:
        q_value += 0.02
    
    # Check for progress indicators in next_state
    progress_patterns = [
        r'created',
        r'written',
        r'updated',
        r'changed',
        r'modified',
        r'added',
        r'copied',
        r'moved',
        r'saved',
        r'extracted',
        r'decoded',
        r'encrypted',
    ]
    
    progress_count = 0
    for pattern in progress_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            progress_count += 1
    
    q_value += 0.03 * min(progress_count, 3)
    
    # Penalize if action is empty or whitespace only
    if not action_stripped:
        q_value -= 0.1
    
    # Clamp value between 0.0 and 1.0
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value