import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for terminal success states
    success_indicators = ['success', 'done', 'completed', 'achieved', 'finished']
    if any(indicator in state_lower for indicator in success_indicators):
        return 1.0
    
    # Check for terminal failure states
    failure_indicators = ['failed', 'impossible', 'timeout', 'error', 'limit reached']
    if any(indicator in state_lower for indicator in failure_indicators):
        return 0.0
    
    # Extract steps taken if available in state
    steps_taken = 0
    step_match = re.search(r'(\d+)\s*steps?', state_lower)
    if step_match:
        try:
            steps_taken = int(step_match.group(1))
        except:
            steps_taken = 0
    
    # Count task-progress indicators in the state
    # These are common action verbs and location prepositions in ALFWorld
    task_indicators = [
        'put', 'move', 'take', 'go', 'navigate', 'clean', 'heat', 'cool',
        'open', 'close', 'turn', 'in', 'on', 'at', 'under', 'to'
    ]
    
    indicator_count = sum(1 for indicator in task_indicators if indicator in state_lower)
    
    # Base progress score on number of task indicators
    if indicator_count >= 4:
        base_value = 0.6
    elif indicator_count >= 2:
        base_value = 0.4
    else:
        base_value = 0.2
    
    # Apply step penalty (fewer steps = higher value, max 40 steps)
    if steps_taken > 0:
        step_ratio = steps_taken / 40.0
        base_value = base_value * (1.0 - step_ratio * 0.3)
    
    # Small bonus if goal or task is explicitly mentioned
    if 'goal' in state_lower or 'task' in state_lower:
        base_value = min(0.95, base_value + 0.05)
    
    # Ensure value is in valid range [0.0, 1.0]
    return min(1.0, max(0.0, base_value))