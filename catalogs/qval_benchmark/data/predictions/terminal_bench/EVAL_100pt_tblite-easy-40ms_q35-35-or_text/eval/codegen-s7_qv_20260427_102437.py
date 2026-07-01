import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.5
    
    success_patterns = [
        r'\bsuccess\b', r'\bpassed\b', r'\bverified\b', 
        r'\bcomplete\b', r'\bdone\b', r'\bcorrect\b',
        r'\bverification passed\b', r'\btest passed\b'
    ]
    
    error_patterns = [
        r'\berror\b', r'\bfail\b', r'\bexception\b', 
        r'\btraceback\b', r'\bdenied\b', r'\bnot found\b',
        r'\bpermission\b', r'\bsegmentation fault\b', r'\bcrash\b',
        r'\btimeout\b', r'\bfailed\b'
    ]
    
    next_lower = next_state.lower()
    state_lower = state.lower()
    
    has_success = any(re.search(p, next_lower) for p in success_patterns)
    has_error = any(re.search(p, next_lower) for p in error_patterns)
    has_prev_error = any(re.search(p, state_lower) for p in error_patterns)
    
    if has_success:
        q_value = 0.95
    elif has_error:
        q_value = 0.05
    elif has_prev_error:
        q_value = 0.2
    else:
        q_value = 0.5
        
    action_lower = action.lower()
    
    if re.search(r'\brm -rf\s+/', action_lower):
        q_value = 0.0
        
    if re.search(r'\bsubmit\b|\bexit\b|\bfinish\b', action_lower):
        if not has_error:
            q_value = max(q_value, 0.8)
            
    if q_value < 0.0:
        q_value = 0.0
    elif q_value > 1.0:
        q_value = 1.0
        
    return float(q_value)