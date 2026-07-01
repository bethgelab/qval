import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value estimate
    q_value = 0.5
    
    # Success indicators in next_state
    success_patterns = [
        r'success', r'passed', r'verified', r'correct', r'accepted',
        r'flag\{', r'answer', r'complete', r'congratulations', r'solution'
    ]
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.4
            break
    
    # Error indicators in next_state
    error_patterns = [
        r'error', r'failed', r'exception', r'permission denied',
        r'not found', r'syntax error', r'no such', r'invalid', r'failure'
    ]
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value -= 0.5
            break
            
    # Action quality analysis
    action_lower = action.lower()
    
    # Constructive actions
    if any(k in action_lower for k in ['submit', 'verify', 'check', 'run', 'solve', 'execute', 'python', 'bash']):
        q_value += 0.1
        
    # Destructive actions
    if any(k in action_lower for k in ['rm -rf', 'dd ', 'mkfs', 'format', 'reboot', 'shutdown', 'kill -9']):
        q_value -= 0.3
        
    # Progress indicator (next_state larger than state implies output)
    if len(next_state) > len(state):
        q_value += 0.05
        
    # Clamp Q-value between 0.0 and 1.0
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value