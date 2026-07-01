import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    score = 0.5
    
    positive_patterns = [
        r'success', r'correct', r'flag', r'solution', r'verified', 
        r'passed', r'congratulations', r'accepted', r'output'
    ]
    
    negative_patterns = [
        r'error', r'failed', r'permission denied', r'not found', 
        r'syntax error', r'exception', r'undefined', r'invalid'
    ]
    
    completion_actions = [
        r'submit', r'check', r'verify', r'answer', r'finish'
    ]
    
    destructive_actions = [
        r'rm -rf', r'reboot', r'sudo rm', r'format'
    ]
    
    info_actions = [r'ls', r'cat', r'grep', r'find', r'head', r'pwd']
    
    # Check for positive signals in next_state
    for pattern in positive_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            score += 0.4
            break
            
    # Check for negative signals in next_state
    for pattern in negative_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            score -= 0.4
            break
            
    # Check action type for completion
    for pattern in completion_actions:
        if re.search(pattern, action, re.IGNORECASE):
            score += 0.2
            break
            
    # Check action type for destruction
    for pattern in destructive_actions:
        if re.search(pattern, action, re.IGNORECASE):
            score -= 0.5
            break
            
    # Check action type for information gathering
    for pattern in info_actions:
        if re.search(pattern, action, re.IGNORECASE):
            score += 0.1
            break
            
    return max(0.0, min(1.0, score))