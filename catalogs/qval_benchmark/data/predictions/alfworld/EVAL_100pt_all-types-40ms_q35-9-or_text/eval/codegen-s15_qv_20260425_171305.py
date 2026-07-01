import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    # Check for goal completion signals in next state
    completion_patterns = [
        r'done', r'success', r'completed', r'goal reached', 
        r'task done', r'finished', r'goal achieved'
    ]
    for pattern in completion_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value += 0.6
            break
    
    # Check for failure/error signals in next state
    failure_patterns = [
        r'error', r'failed', r'cannot', r'invalid', 
        r'not allowed', r'not possible', r'already', r'empty'
    ]
    for pattern in failure_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value -= 0.5
            break
    
    # Check for positive action effects
    positive_effects = [
        r'clean', r'picked up', r'put down', r'placed', 
        r'found', r'opened', r'closed', r'cleaned', 
        r'washed', r'dried', r'sorted'
    ]
    for effect in positive_effects:
        if re.search(effect, next_state, re.IGNORECASE):
            q_value += 0.15
            break
    
    # Check for negative state changes
    negative_effects = [
        r'dirty', r'broken', r'lost', r'missing', 
        r'stuck', r'blocked', r'empty'
    ]
    for effect in negative_effects:
        if re.search(effect, next_state, re.IGNORECASE):
            q_value -= 0.1
            break
    
    # Bonus if state changed (action had effect)
    if state != next_state:
        q_value += 0.1
    
    # Check for location-related progress
    location_keywords = ['kitchen', 'bedroom', 'bathroom', 'living room', 'hallway']
    if re.search('|'.join(location_keywords), next_state, re.IGNORECASE):
        q_value += 0.05
    
    # Cap Q-value to reasonable range
    q_value = max(-0.8, min(1.0, q_value))
    
    return q_value