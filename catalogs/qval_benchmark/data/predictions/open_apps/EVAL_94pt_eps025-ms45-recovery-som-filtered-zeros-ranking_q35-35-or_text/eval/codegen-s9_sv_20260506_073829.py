import re

def signal_function(state: str) -> float:
    s = state.lower()
    
    # Failure indicators: High priority penalty
    failure_patterns = [
        r'\berror\b', r'\binvalid\b', r'\b404\b', r'\bfailed\b', r'\bexception\b',
        r'\bnot found\b', r'\bcannot\b', r'\brequired\b', r'\btry again\b'
    ]
    for pattern in failure_patterns:
        if re.search(pattern, s):
            return 0.05

    # Success indicators: High priority reward
    success_patterns = [
        r'\bsaved\b', r'\bsent\b', r'\badded\b', r'\bcreated\b', r'\bcompleted\b',
        r'\bsuccess\b', r'\bdone\b', r'\bevent added\b', r'\bmessage sent\b',
        r'\btask completed\b', r'\bupdated\b', r'\bposted\b', r'\bpublished\b'
    ]
    for pattern in success_patterns:
        if re.search(pattern, s):
            return 0.95

    # Step count analysis for efficiency (if available in state)
    step_score = 1.0
    step_match = re.search(r'\bstep\s*(\d+)\b', s)
    if step_match:
        step_num = int(step_match.group(1))
        if step_num > 35:
            step_score = 0.5
        elif step_num > 25:
            step_score = 0.75

    # Progress indicators: App context + Action verbs
    app_keywords = [r'\bcalendar\b', r'\btodo\b', r'\bmessenger\b', r'\bmaps\b', r'\bcode editor\b']
    action_keywords = [r'\bsave\b', r'\bsend\b', r'\bcreate\b', r'\bnew\b', r'\bedit\b', r'\bform\b']

    app_count = sum(1 for p in app_keywords if re.search(p, s))
    action_count = sum(1 for p in action_keywords if re.search(p, s))

    if app_count > 0 and action_count > 0:
        return 0.6 * step_score
    elif app_count > 0:
        return 0.4 * step_score
    elif action_count > 0:
        return 0.3 * step_score

    return 0.1