import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value estimate
    q_value = 0.0
    
    # Check for completion/success signals in next_state
    completion_patterns = [
        r'success', r'done', r'completed', r'passed', r'finished',
        r'\u2713', r'\u2714', r'\bOK\b', r'\bok\b', r'\btrue\b', r'\bTrue\b',
        r'verification.*pass', r'correct', r'valid', r'\.0$', r'exit code 0',
        r'exit_code.*0', r'return.*0', r'no error', r'all tests passed'
    ]
    
    # Check for error/failure signals in next_state
    error_patterns = [
        r'\berror\b', r'\bfail\b', r'\bfailed\b', r'\bexception\b', r'\bError\b',
        r'\bException\b', r'not found', r'permission denied', r'syntax error',
        r'\binvalid\b', r'\u2717', r'\u2718', r'\bfalse\b', r'\bFalse\b',
        r'exit code [1-9]', r'exit_code.*[1-9]', r'return.*[1-9]',
        r'FATAL', r'\btimeout\b', r'\binterrupted\b'
    ]
    
    # Check for progress indicators
    progress_patterns = [
        r'created', r'wrote', r'saved', r'generated', r'built',
        r'installed', r'copied', r'moved', r'downloaded', r'extracted',
        r'compiled', r'executed', r'ran', r'processed', r'updated',
        r'wrote.*bytes', r'written', r'successfully'
    ]
    
    # Analyze next_state for signals
    next_state_lower = next_state.lower()
    
    # Count completion signals
    completion_count = sum(1 for pattern in completion_patterns 
                          if re.search(pattern, next_state_lower, re.IGNORECASE))
    
    # Count error signals
    error_count = sum(1 for pattern in error_patterns 
                     if re.search(pattern, next_state_lower, re.IGNORECASE))
    
    # Count progress signals
    progress_count = sum(1 for pattern in progress_patterns 
                        if re.search(pattern, next_state_lower, re.IGNORECASE))
    
    # Strong completion indicators (high confidence of task success)
    strong_completion = any([
        re.search(r'verification.*pass', next_state_lower, re.IGNORECASE),
        re.search(r'all tests passed', next_state_lower, re.IGNORECASE),
        re.search(r'\.0$', next_state_lower),
        re.search(r'exit code 0', next_state_lower, re.IGNORECASE)
    ])
    
    # Update Q-value based on signals
    if strong_completion:
        q_value = 0.95
    elif completion_count > 0:
        q_value = 0.7 + min(0.2, completion_count * 0.05)
    elif error_count > 0:
        q_value = -0.5 - min(0.3, error_count * 0.1)
    elif progress_count > 0:
        q_value = 0.3 + min(0.3, progress_count * 0.08)
    
    # Action efficiency bonus (prefer concise, targeted actions)
    action_tokens = action.strip().split()
    if len(action_tokens) <= 3 and len(action_tokens) > 0:
        q_value += 0.05
    elif len(action_tokens) > 10:
        q_value -= 0.1
    
    # State change magnitude (indicates meaningful progress)
    state_change = abs(len(next_state) - len(state))
    if state_change > 50:
        q_value += min(0.1, state_change / 5000)
    
    # Check for specific file creation/verification patterns
    if re.search(r'\.(txt|py|json|yaml|yml|csv|log|out)$', next_state):
        q_value += 0.05
    
    # Penalize if state appears stuck (minimal change)
    if state_change < 10 and completion_count == 0 and progress_count == 0:
        q_value -= 0.1
    
    # Clamp Q-value to reasonable range [-1.0, 1.0]
    q_value = max(-1.0, min(1.0, q_value))
    
    return float(q_value)