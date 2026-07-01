import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Outcome keywords indicating task completion (Success)
    success_keywords = [
        r'\bsuccess\b', r'\bcompleted\b', r'\bcreated\b', r'\bsaved\b',
        r'\bsent\b', r'\badded\b', r'\bconfirmed\b', r'\bdone\b',
        r'\bevent created\b', r'\bmessage sent\b', r'\btodo added\b',
        r'\btask completed\b', r'\blocation saved\b', r'\brun successful\b'
    ]
    
    # Outcome keywords indicating failure or error
    error_keywords = [
        r'\berror\b', r'\bfailed\b', r'\binvalid\b', r'\b404\b',
        r'\bunavailable\b', r'\btimeout\b', r'\bnot found\b'
    ]
    
    # Action keywords indicating a finalizing step
    finalizing_actions = [
        r'\bsubmit\b', r'\bsend\b', r'\bsave\b', r'\bcreate\b', r'\badd\b',
        r'\bconfirm\b', r'\baccept\b'
    ]
    
    # Check for errors in next_state first (overrides success)
    for pattern in error_keywords:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.0
            
    # Check for immediate success in next_state
    for pattern in success_keywords:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 1.0
            
    # If no immediate success or error, estimate based on action and context
    q_score = 0.1
    
    # Check if action looks like a goal-completing step
    is_finalizing = False
    for pattern in finalizing_actions:
        if re.search(pattern, action, re.IGNORECASE):
            is_finalizing = True
            break
            
    # Check if next_state shows progress (e.g. new page content)
    progress_indicators = [
        r'\binbox\b', r'\bevent\b', r'\btodo\b', r'\bmessage\b',
        r'\bmap\b', r'\bcode\b', r'\blist\b', r'\bresults\b'
    ]
    has_progress = any(re.search(p, next_state, re.IGNORECASE) for p in progress_indicators)
    
    if is_finalizing:
        # High probability if action is finalizing and no error
        q_score = 0.8
    elif re.search(r'\bfill\b', action, re.IGNORECASE):
        # Form filling is progress
        q_score = 0.4
    elif re.search(r'\bpress\b', action, re.IGNORECASE):
        # Pressing Enter often submits
        q_score = 0.5
    elif re.search(r'\bclick\b', action, re.IGNORECASE):
        # Clicking links/buttons is progress
        q_score = 0.4
    elif re.search(r'\bscroll\b', action, re.IGNORECASE):
        # Scrolling is low progress
        q_score = 0.2
    elif re.search(r'\bnoop\b', action, re.IGNORECASE):
        # No operation is low progress
        q_score = 0.05
        
    # Boost score if next_state shows relevant content
    if has_progress:
        q_score = max(q_score, 0.3)
        
    # Clamp values to [0.0, 1.0]
    if q_score > 1.0:
        q_score = 1.0
    elif q_score < 0.0:
        q_score = 0.0
        
    return float(q_score)