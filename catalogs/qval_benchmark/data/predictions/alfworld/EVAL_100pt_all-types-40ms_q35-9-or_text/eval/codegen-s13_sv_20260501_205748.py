def signal_function(state: str) -> float:
    import re
    
    value = 0.0
    
    # Check if goal is already completed
    if re.search(r'(goal|task).*completed|done|success', state, re.IGNORECASE):
        return 0.98
    
    # Check for failure/error states
    if re.search(r'(error|fail|impossible|cannot|blocked|unable)', state, re.IGNORECASE):
        return -0.3
    
    # Check if agent is at the correct location for the goal
    if re.search(r'at\s+(?:kitchen|bedroom|bathroom|living room|dining room)', state, re.IGNORECASE):
        value = 0.6
    else:
        value = 0.4
    
    # Check for object states that affect success
    if re.search(r'(clean|fresh|dry|washed)\s+(?:towel|cup|bowl|plate|glass)', state, re.IGNORECASE):
        value = max(value, 0.7)
    
    if re.search(r'(dirty|soiled|wet|stained)\s+(?:towel|cup|bowl|plate|glass)', state, re.IGNORECASE):
        value = min(value, 0.5)
    
    # Check for action mentions in state (indicates progress)
    action_keywords = ['pick', 'clean', 'move', 'carry', 'put', 'drop', 'place', 'take']
    action_count = len(re.findall(r'\b(' + '|'.join(action_keywords) + r')\b', state, re.IGNORECASE))
    
    if action_count >= 3:
        value = max(value, 0.6)
    elif action_count >= 1:
        value = max(value, 0.4)
    
    # Check for goal mention (task is defined)
    if re.search(r'(goal|task)', state, re.IGNORECASE):
        value = max(value, 0.3)
    
    # Cap the value between -0.5 and 1.0
    return max(-0.5, min(1.0, value))