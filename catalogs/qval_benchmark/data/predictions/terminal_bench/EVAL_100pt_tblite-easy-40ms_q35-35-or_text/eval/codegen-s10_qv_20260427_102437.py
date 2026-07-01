import re

def signal_function(state: str, action: str, next_state: str) -> float:
    score = 0.5
    
    next_lower = next_state.lower()
    action_lower = action.lower()
    
    # Success indicators (High Q)
    success_patterns = [
        r'\bsuccess\b', r'\bpassed\b', r'\bverified\b', 
        r'\bcomplete\b', r'\bdone\b', r'\bresult\b', 
        r'\boptimal\b', r'\bcorrect\b', r'\b100%\b', r'\b1\.0\b',
        r'\bverification passed\b', r'\btest passed\b'
    ]
    for pattern in success_patterns:
        if re.search(pattern, next_lower):
            score += 0.35
            break
            
    # Error indicators (Low Q)
    error_patterns = [
        r'\berror\b', r'\bfailed\b', r'\bexception\b', 
        r'\btraceback\b', r'\bdenied\b', r'\bnot found\b', 
        r'\bmissing\b', r'\bfailure\b', r'\bundefined\b', 
        r'\btimeout\b', r'\bkill\b', r'\bpermission denied\b'
    ]
    for pattern in error_patterns:
        if re.search(pattern, next_lower):
            score -= 0.6
            break
            
    # Destructive action check (Safety)
    destructive_patterns = [
        r'rm\s+-rf\s+/', r'sudo\s+rm\s+-rf\s+/', r'chmod\s+-R\s+000'
    ]
    for pattern in destructive_patterns:
        if re.search(pattern, action_lower):
            score -= 0.5
            break
            
    # Stagnation check (No progress)
    if state.strip() == next_state.strip():
        score -= 0.2
        
    # Clamp score between 0.0 and 1.0
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
        
    return float(score)