import re
import math

def signal_function(state: str) -> float:
    if not state or not isinstance(state, str):
        return 0.0
    
    score = 0.0
    
    # 1. Check for Immediate Success (Reward = 1.0)
    success_patterns = [
        r'success', r'completed', r'goal achieved', r'task complete',
        r'confirmed', r'saved', r'added', r'sent', r'created'
    ]
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 1.0
            
    # 2. Check for Immediate Failure (Reward = 0.0)
    failure_patterns = [
        r'404', r'not found', r'critical error', r'connection refused',
        r'authentication failed', r'blocked'
    ]
    for pattern in failure_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 0.0
            
    # 3. Estimate Progress towards Goal
    # A. Form Completion: Check for filled input fields
    filled_count = 0
    filled_patterns = [
        r'role="textbox".*?value="[^"]+"',
        r'role="input".*?value="[^"]+"',
        r'role="text".*?value="[^"]+"'
    ]
    for pattern in filled_patterns:
        filled_count += len(re.findall(pattern, state, re.IGNORECASE | re.DOTALL))
    score += min(0.4, filled_count * 0.1)
    
    # B. Action Availability: Check for submit/save buttons
    action_keywords = ['save', 'submit', 'add', 'send', 'create', 'confirm', 'done', 'next']
    has_action = False
    for keyword in action_keywords:
        if re.search(r'role="button".*?' + keyword, state, re.IGNORECASE | re.DOTALL):
            has_action = True
            break
    if has_action:
        score += 0.3
        
    # C. Navigation Context: Check if we are on a relevant page
    if re.search(r'role="link".*?(home|back|next|calendar|todo|messenger|maps|code)', state, re.IGNORECASE | re.DOTALL):
        score += 0.1
        
    # 4. Penalties for State Issues
    # A. Loading States (Delays progress)
    if re.search(r'loading|processing|please wait|spinner|fetching', state, re.IGNORECASE):
        score -= 0.3
        
    # B. Warnings/Soft Errors
    if re.search(r'warning|invalid|error|failed', state, re.IGNORECASE):
        score -= 0.2
        
    # 5. Clamp Score to [0.0, 1.0]
    return max(0.0, min(1.0, score))