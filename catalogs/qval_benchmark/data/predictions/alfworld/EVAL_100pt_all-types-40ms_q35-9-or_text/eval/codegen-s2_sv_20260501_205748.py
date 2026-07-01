import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    completion_patterns = [
        r'done', r'completed', r'success', r'task finished',
        r'goal achieved', r'object placed', r'target reached',
        r'task complete', r'object in', r'placed on'
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, state_lower):
            return 1.0
    
    failure_patterns = [
        r'blocked', r'cannot', r'error', r'failed',
        r'not possible', r'impossible', r'cannot reach',
        r'out of range', r'no path', r'obstacle'
    ]
    
    for pattern in failure_patterns:
        if re.search(pattern, state_lower):
            return 0.0
    
    step_match = re.search(r'(\d+)', state_lower)
    if step_match:
        steps_remaining = int(step_match.group(1))
        if steps_remaining >= 35:
            return 0.2
        elif steps_remaining >= 25:
            return 0.35
        elif steps_remaining >= 15:
            return 0.5
        elif steps_remaining >= 10:
            return 0.65
        elif steps_remaining >= 5:
            return 0.75
        else:
            return 0.85
    
    progress_keywords = [
        r'in hand', r'holding', r'in room', r'located',
        r'pickup', r'place', r'move', r'clean',
        r'pick up', r'put on', r'place on'
    ]
    
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    
    task_indicators = [
        r'task', r'goal', r'objective', r'object', r'room',
        r'location', r'place', r'desk', r'floor', r'shelf',
        r'chair', r'table', r'counter', r'kitchen', r'bedroom'
    ]
    
    task_count = sum(1 for kw in task_indicators if kw in state_lower)
    
    if progress_count >= 3:
        return 0.6
    elif progress_count >= 2:
        return 0.45
    elif progress_count >= 1:
        return 0.35
    elif task_count >= 3:
        return 0.4
    elif task_count >= 2:
        return 0.3
    elif task_count >= 1:
        return 0.2
    
    return 0.15