def signal_function(state: str) -> float:
    import re
    
    # Check for success/completion indicators - highest value
    success_patterns = [
        r'success',
        r'task completed',
        r'task complete',
        r'you have completed',
        r'thanks for playing',
        r'episode finished',
        r'task done',
        r'congratulations',
        r'well done',
        r'task accomplished'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 1.0
    
    # Check for failure/terminal failure indicators - lowest value
    failure_patterns = [
        r'failed',
        r'fail',
        r'too many steps',
        r'max steps reached',
        r'episode terminated',
        r'out of steps',
        r'time out'
    ]
    
    for pattern in failure_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 0.0
    
    # Base value for an active episode in progress
    value = 0.3
    
    # Bonus for having picked up an object (inventory state)
    if re.search(r'you are holding|holding:', state, re.IGNORECASE):
        value += 0.2
    
    # Bonus for placing/putting an object (task progress indicator)
    if re.search(r'you have put|you placed|put.*on|put.*in|placed.*on|placed.*in', state, re.IGNORECASE):
        value += 0.15
    
    # Bonus for location awareness (navigation progress)
    if re.search(r'you are in|you are at|looking at|you see', state, re.IGNORECASE):
        value += 0.1
    
    # Count available meaningful actions (more options = better state)
    action_keywords = ['go to', 'look', 'open', 'close', 'take', 'put', 'clean', 'heat', 'cool', 'toggle', 'use', 'break', 'move']
    action_count = sum(1 for keyword in action_keywords if keyword.lower() in state.lower())
    
    if action_count >= 5:
        value += 0.1
    elif action_count >= 3:
        value += 0.05
    
    # Bonus for task-related keywords (indicates goal-directed activity)
    task_keywords = ['find', 'locate', 'search', 'put', 'take', 'clean', 'heat', 'cool', 'place', 'bring', 'move']
    task_match_count = sum(1 for keyword in task_keywords if keyword.lower() in state.lower())
    
    if task_match_count >= 3:
        value += 0.1
    elif task_match_count >= 1:
        value += 0.05
    
    # Penalize states with error messages
    error_keywords = ['cannot', 'not found', 'does not exist', 'invalid', 'error', 'nothing to', 'no such']
    error_count = sum(1 for keyword in error_keywords if keyword.lower() in state.lower())
    
    if error_count > 0:
        value -= 0.08 * min(error_count, 2)
    
    # Penalize repeated failed attempts
    if re.search(r'try again|not possible|impossible', state, re.IGNORECASE):
        value -= 0.1
    
    # Ensure value is in valid range [0, 1]
    value = max(0.0, min(1.0, value))
    
    return value