import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Success indicators in the next state (Goal Achieved)
    success_patterns = [
        r'(?i)success',
        r'(?i)saved',
        r'(?i)added',
        r'(?i)sent',
        r'(?i)created',
        r'(?i)completed',
        r'(?i)done',
        r'(?i)event.*added',
        r'(?i)task.*added',
        r'(?i)message.*sent'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state):
            return 1.0
            
    # Error indicators in the next state (Negative Outcome)
    error_patterns = [
        r'(?i)error',
        r'(?i)fail',
        r'(?i)invalid',
        r'(?i)failed'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, next_state):
            return 0.0
            
    # Progress indicators based on action (High Confidence)
    progress_patterns = [
        r'click.*[sS]ubmit',
        r'click.*[sS]ave',
        r'click.*[sS]end',
        r'click.*[aA]dd',
        r'click.*[nN]ew',
        r'fill.*[tT]ext'
    ]
    
    for pattern in progress_patterns:
        if re.search(pattern, action):
            return 0.5
            
    # Navigation or general actions (Low Confidence)
    if re.search(r'click', action):
        return 0.2
        
    if re.search(r'scroll', action):
        return 0.1
        
    if re.search(r'noop', action):
        return 0.0
        
    return 0.0