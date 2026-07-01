def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base score starts at 0.5 (neutral expectation)
    score = 0.5
    
    # Analyze the action - is it a reasonable command?
    action_lower = action.lower() if action else ""
    
    # Good command patterns for terminal tasks
    good_commands = ['cd', 'ls', 'cat', 'echo', 'mkdir', 'touch', 'cp', 'mv', 'rm', 
                     'find', 'grep', 'sed', 'awk', 'python', 'bash', 'sh', 'chmod',
                     'tar', 'zip', 'unzip', 'git', 'ssh', 'scp', 'curl', 'wget',
                     'head', 'tail', 'wc', 'sort', 'uniq', 'diff', 'patch',
                     'make', 'gcc', 'g++', 'pip', 'install', 'run', 'test',
                     'verify', 'check', 'validate', 'submit', 'print', 'read']
    
    # Score based on command quality
    action_score = 0.0
    for cmd in good_commands:
        if cmd in action_lower:
            action_score += 0.05
    
    # Penalize obviously destructive actions
    bad_patterns = ['rm -rf /', 'dd if=/dev/zero', 'mkfs', 'format c:', 'wipe',
                    '> /dev/sda', 'chmod 777 /', 'shutdown', 'reboot']
    for bad in bad_patterns:
        if bad in action_lower:
            action_score -= 0.3
    
    score += action_score
    
    # Analyze state progression
    state_lines = len(state.split('\n')) if state else 0
    next_state_lines = len(next_state.split('\n')) if next_state else 0
    
    # Progress indicator - more output often means progress
    if next_state_lines > state_lines:
        score += 0.08
    elif next_state_lines < state_lines:
        score -= 0.03
    
    # Check for error indicators in states
    error_patterns = ['error', 'failed', 'permission denied', 'not found', 
                      'no such file', 'cannot', 'unsuccessful', 'invalid',
                      'traceback', 'exception', 'failed to']
    
    state_errors = sum(1 for pattern in error_patterns if pattern in state.lower())
    next_errors = sum(1 for pattern in error_patterns if pattern in next_state.lower())
    
    if next_errors > state_errors:
        score -= 0.15
    elif next_errors < state_errors:
        score += 0.10
    elif next_errors > 0:
        score -= 0.05
    
    # Check for success indicators
    success_patterns = ['success', 'completed', 'done', 'finished', 'passed', 
                        'verified', 'correct', 'matched', 'ok', 'true',
                        'test passed', 'verification', 'solution', 'answer']
    
    state_success = sum(1 for pattern in success_patterns if pattern in state.lower())
    next_success = sum(1 for pattern in success_patterns if pattern in next_state.lower())
    
    if next_success > state_success:
        score += 0.20
    elif next_success > 0:
        score += 0.05
    
    # Check for file/directory creation (progress)
    creation_patterns = ['created', 'copied', 'moved', 'extracted', 'downloaded',
                         'installed', 'generated', 'written']
    if any(pattern in next_state.lower() for pattern in creation_patterns):
        score += 0.10
    
    # Penalize if action appears to be empty or just whitespace
    if not action or action.strip() == '':
        score -= 0.15
    
    # Penalize if action is just a prompt character
    if action and action.strip() in ['$', '#', '%', '>']:
        score -= 0.10
    
    # Check for terminal task completion hints
    completion_hints = ['task complete', 'all tests passed', 'final', 'result',
                        'output', 'answer is', 'solution found']
    if any(hint in next_state.lower() for hint in completion_hints):
        score += 0.25
    
    # Clamp score to valid probability range
    return max(0.0, min(1.0, score))