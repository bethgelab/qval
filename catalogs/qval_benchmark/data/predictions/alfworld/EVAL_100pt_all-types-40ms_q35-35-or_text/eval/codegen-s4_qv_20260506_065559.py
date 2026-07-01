import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for terminal success
    if re.search(r'\b(success|goal accomplished|task complete|done|reached)\b', next_state, re.I):
        return 1.0
    
    # Check for terminal failure
    if re.search(r'\b(fail|error|invalid|cannot|stuck)\b', next_state, re.I):
        return 0.0
    
    # Check for exact state match (action had no effect)
    if state == next_state:
        return 0.0
    
    action_lower = action.lower()
    next_lower = next_state.lower()
    
    score = 0.0
    
    # Evaluate Action-Outcome Consistency
    if 'put' in action_lower:
        # Check if object is placed on a surface
        if re.search(r'\bon\s+\w+', next_lower) or 'placed' in next_lower:
            score = 0.8
        else:
            score = 0.3
    elif 'take' in action_lower:
        # Check if object is held
        if 'holding' in next_lower or 'in your hand' in next_lower:
            score = 0.7
        else:
            score = 0.3
    elif 'clean' in action_lower or 'wash' in action_lower:
        # Check if object is clean
        if 'clean' in next_lower:
            score = 0.8
        else:
            score = 0.3
    elif 'turn on' in action_lower:
        # Check if object is on
        if 'on' in next_lower:
            score = 0.7
        else:
            score = 0.3
    elif 'go to' in action_lower:
        # Navigation progress
        if 'in the' in next_lower:
            score = 0.4
        else:
            score = 0.2
    else:
        # Generic action
        if state != next_state:
            score = 0.2
        else:
            score = 0.0
    
    return max(0.0, min(1.0, score))