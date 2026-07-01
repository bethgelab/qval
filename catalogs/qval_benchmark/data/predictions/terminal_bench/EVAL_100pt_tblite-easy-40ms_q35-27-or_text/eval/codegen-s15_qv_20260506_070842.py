import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value estimate (neutral starting point)
    q_value = 0.5
    
    # Positive indicators suggesting progress toward goal completion
    positive_patterns = [
        r'\bsuccess\b', r'\bcompleted\b', r'\bdone\b', r'\bpassed\b',
        r'\bverified\b', r'\bcorrect\b', r'\bvalid\b', r'\btrue\b',
        r'\bfinished\b', r'\bready\b', r'\bOK\b', r'\baccepted\b',
        r'\bcreated\b', r'\bgenerated\b', r'\bexecuted\b', r'\boutput\b',
        r'\bresult\b', r'\bfound\b', r'\bmatch\b', r'\bexists\b',
        r'\bcomplete\b', r'\bsolved\b', r'\bcorrectly\b'
    ]
    
    # Negative indicators suggesting problems or failure
    negative_patterns = [
        r'\berror\b', r'\bfail\b', r'\bexception\b', r'\bwarning\b',
        r'\bdenied\b', r'\binvalid\b', r'\bfalse\b', r'\btimeout\b',
        r'\bmissing\b', r'\bnot found\b', r'\bpermission\b', r'\brefused\b',
        r'\bfailed\b', r'\bwrong\b', r'\bincorrect\b', r'\btraceback\b',
        r'\bunauthorized\b', r'\bdenied\b', r'\bno such\b'
    ]
    
    # Normalize text for pattern matching
    state_text = state.lower()
    next_state_text = next_state.lower()
    action_text = action.lower()
    
    # Count positive indicators
    state_positive = sum(1 for p in positive_patterns if re.search(p, state_text))
    next_positive = sum(1 for p in positive_patterns if re.search(p, next_state_text))
    
    # Count negative indicators
    state_negative = sum(1 for p in negative_patterns if re.search(p, state_text))
    next_negative = sum(1 for p in negative_patterns if re.search(p, next_state_text))
    
    # Reward for positive indicators in next state
    q_value += next_positive * 0.12
    
    # Penalty for negative indicators in next state
    q_value -= next_negative * 0.18
    
    # Reward for progress (more positive or fewer negative indicators)
    if next_positive > state_positive:
        q_value += 0.08
    if next_negative < state_negative:
        q_value += 0.06
    
    # Strong reward for explicit completion markers
    completion_patterns = [r'\bcompleted\b', r'\bsuccess\b', r'\bpassed\b', r'\bsolved\b', r'\bverified\b']
    if any(re.search(p, next_state_text) for p in completion_patterns):
        q_value += 0.25
    
    # Strong penalty for explicit failure markers
    failure_patterns = [r'\bfailed\b', r'\berror\b', r'\bexception\b', r'\btraceback\b']
    if any(re.search(p, next_state_text) for p in failure_patterns):
        q_value -= 0.25
    
    # Check for step/iteration indicators (fewer remaining steps = higher value)
    step_match = re.search(r'\b(step|iteration|epoch|round)\s*[:\d]*\s*(\d+)', next_state_text)
    if step_match:
        step_num = int(step_match.group(2))
        # Assuming 40-step limit, reward being early
        if step_num <= 20:
            q_value += 0.1
        elif step_num <= 30:
            q_value += 0.05
    
    # Cap the value between 0 and 1
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value