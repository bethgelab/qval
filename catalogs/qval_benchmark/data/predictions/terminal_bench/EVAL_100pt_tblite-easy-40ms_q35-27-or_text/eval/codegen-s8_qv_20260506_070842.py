import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value starts at 0.5 (neutral expectation)
    q_value = 0.5
    
    # Normalize for analysis
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Success/completion indicators - strong positive signal
    success_patterns = ['success', 'complete', 'done', 'finished', 'ok', 'passed', 
                        'verified', 'correct', 'match', 'found', 'exists', 'true']
    for pattern in success_patterns:
        if pattern in next_state_lower:
            q_value += 0.12
    
    # Error/failure indicators - strong negative signal
    error_patterns = ['error', 'fail', 'failed', 'permission denied', 'not found', 
                      'cannot', 'refused', 'invalid', 'exception', 'traceback', 
                      'denied', 'abort', 'killed', 'timed out']
    for pattern in error_patterns:
        if pattern in next_state_lower:
            q_value -= 0.15
    
    # Task progress indicators - moderate positive signal
    progress_patterns = ['created', 'wrote', 'saved', 'generated', 'output', 
                         'result', 'computed', 'calculated', 'processed', 
                         'extracted', 'downloaded', 'uploaded', 'copied', 'moved']
    for pattern in progress_patterns:
        if pattern in next_state_lower:
            q_value += 0.06
    
    # Action effectiveness - productive commands
    productive_commands = ['cat', 'grep', 'find', 'ls', 'mkdir', 'cp', 'mv', 
                           'rm', 'touch', 'echo', 'python', 'bash', 'chmod', 
                           'chown', 'sed', 'awk', 'tar', 'zip', 'unzip', 'wget', 
                           'curl', 'ssh', 'scp', 'git', 'pip', 'conda', 'make']
    for cmd in productive_commands:
        if action_lower.startswith(cmd) or ' ' + cmd + ' ' in action_lower:
            q_value += 0.03
    
    # Detect if next state has new information (progress indicator)
    state_lines = set(line.strip() for line in state.split('\n') if line.strip())
    next_state_lines = set(line.strip() for line in next_state.split('\n') if line.strip())
    new_lines = next_state_lines - state_lines
    if new_lines:
        # More new content suggests progress
        q_value += min(0.1, len(new_lines) * 0.01)
    
    # Penalize repeated states (stuck in loop)
    if next_state == state:
        q_value -= 0.2
    
    # Penalize very long outputs without clear success (might be stuck in verbose output)
    if len(next_state) > 20000 and not any(p in next_state_lower for p in success_patterns):
        q_value -= 0.08
    
    # Bonus for short, clean commands (efficiency)
    if len(action) < 50 and not any(p in action_lower for p in error_patterns):
        q_value += 0.02
    
    # Special handling for submit/verify type actions
    if any(submit_word in action_lower for submit_word in ['submit', 'verify', 'test', 'check']):
        # These actions should have higher variance based on outcome
        if any(p in next_state_lower for p in success_patterns):
            q_value += 0.2
        elif any(p in next_state_lower for p in error_patterns):
            q_value -= 0.2
    
    # Clamp to valid Q-value range [0, 1]
    return max(0.0, min(1.0, q_value))