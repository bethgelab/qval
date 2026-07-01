import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    # Check for goal achievement indicators in next_state
    success_patterns = [
        r'\bsaved\b', r'\bcreated\b', r'\badded\b', r'\bsent\b',
        r'\bcompleted\b', r'\bdone\b', r'\bsuccess\b', r'\bupdated\b',
        r'\bsubmitted\b', r'\bconfirmed\b', r'\bfinished\b',
        r'\bitem.*added\b', r'\bevent.*created\b', r'\bmessage.*sent\b',
        r'\btask.*completed\b', r'\bmeeting.*created\b', r'\bappointment.*saved\b'
    ]
    
    next_state_lower = next_state.lower()
    for pattern in success_patterns:
        if re.search(pattern, next_state_lower):
            q_value += 0.5
            break
    
    # Check for error/failure indicators
    error_patterns = [
        r'\berror\b', r'\bfailed\b', r'\binvalid\b', r'\brequired\b',
        r'\bmissing\b', r'\bnot found\b', r'\bunauthorized\b',
        r'\bforbidden\b', r'\bdenied\b', r'\bproblem\b'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, next_state_lower):
            q_value -= 0.4
            break
    
    # Check if state changed meaningfully (progress indicator)
    if state != next_state:
        state_words = set(re.findall(r'\w+', state.lower()))
        next_words = set(re.findall(r'\w+', next_state.lower()))
        
        # New content suggests progress
        new_content = len(next_words - state_words)
        if new_content > 10:
            q_value += 0.2
        elif new_content > 5:
            q_value += 0.1
        
        # Significant navigation (low overlap)
        overlap = len(state_words & next_words)
        total = len(state_words | next_words)
        if total > 0 and overlap / total < 0.5:
            q_value += 0.15
    else:
        # No state change - penalize unless it's a no-op action
        if 'noop' not in action.lower() and 'scroll' not in action.lower():
            q_value -= 0.15
    
    # Action effectiveness bonuses
    action_lower = action.lower()
    if 'click' in action_lower:
        q_value += 0.05
    elif 'fill' in action_lower:
        q_value += 0.1
    elif 'press' in action_lower:
        if 'enter' in action_lower or 'submit' in action_lower:
            q_value += 0.08
        else:
            q_value += 0.03
    
    # Check for form field completion indicators
    form_patterns = ['filled', 'entered', 'typed', 'value=', 'input', 'textarea', 'checkbox', 'radio']
    for pattern in form_patterns:
        if pattern in next_state_lower:
            q_value += 0.05
            break
    
    # Check for navigation progress indicators
    nav_patterns = ['href', 'url', 'location', 'path', 'route', 'page', 'section', 'tab', 'view']
    for pattern in nav_patterns:
        if pattern in next_state_lower:
            q_value += 0.03
            break
    
    # Check for interactive elements availability (more options = more progress)
    if 'bid' in next_state or 'button' in next_state_lower or 'link' in next_state_lower:
        q_value += 0.05
    
    # Cap Q-value between 0 and 2.0
    q_value = max(0.0, min(2.0, q_value))
    
    return q_value