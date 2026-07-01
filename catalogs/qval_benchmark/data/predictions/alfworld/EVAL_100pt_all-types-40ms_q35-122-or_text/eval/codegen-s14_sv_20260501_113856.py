def signal_function(state: str) -> float:
    import re
    
    # Convert state to lowercase for consistent matching
    s = state.lower()
    
    # Check for explicit success indicators
    success_patterns = [
        'you have successfully',
        'task completed',
        'congratulations',
        'task is complete',
        'you win',
        'episode complete'
    ]
    for pattern in success_patterns:
        if pattern in s:
            return 1.0
    
    # Check for explicit failure indicators
    failure_patterns = [
        'fail',
        'cannot',
        'error',
        'not possible',
        'invalid action',
        'you cannot',
        'nothing happens'
    ]
    failure_count = sum(1 for pattern in failure_patterns if pattern in s)
    if failure_count >= 2:
        return 0.0
    
    # Check if agent is holding an object (progress indicator)
    holding_patterns = [
        r'you are holding',
        r'you have .* in hand',
        r'holding .*',
        r'grabbed .*',
        r'took .*'
    ]
    has_object = any(re.search(pattern, s) for pattern in holding_patterns)
    
    # Check if agent has placed object at target location
    placement_patterns = [
        r'put .* on .*',
        r'place .* in .*',
        r'put .* in .*',
        r'placed .* on .*',
        r'placed .* in .*'
    ]
    has_placed = any(re.search(pattern, s) for pattern in placement_patterns)
    
    # Check for room context (navigation progress)
    room_keywords = ['kitchen', 'bedroom', 'living room', 'office', 'garage', 'basement', 'bathroom', 'hallway']
    in_room = sum(1 for room in room_keywords if room in s)
    
    # Check for task-relevant objects mentioned
    task_objects = ['recep', 'counter', 'drawer', 'cabinet', 'microwave', 'fridge', 'sink', 'stove', 'table', 'desk']
    object_context = sum(1 for obj in task_objects if obj in s)
    
    # Extract step information if available
    step_match = re.search(r'step\s+(\d+)', s)
    if step_match:
        steps_taken = int(step_match.group(1))
        step_efficiency = max(0.0, (40 - steps_taken) / 40.0)
    else:
        step_efficiency = 0.5  # Assume mid-episode if no step info
    
    # Calculate estimated value based on progress indicators
    value = 0.0
    
    # Base value from remaining steps (efficiency)
    value += step_efficiency * 0.3
    
    # Bonus for having an object (closer to completion)
    if has_object:
        value += 0.25
    
    # Bonus for successful placement (near completion)
    if has_placed:
        value += 0.35
    
    # Bonus for room context (navigation progress)
    value += min(0.15, in_room * 0.05)
    
    # Bonus for task-relevant object context
    value += min(0.1, object_context * 0.02)
    
    # Small penalty for repeated failure patterns
    value -= min(0.1, failure_count * 0.03)
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, value))