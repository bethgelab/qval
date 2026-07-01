def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for ALFWorld environment based on state analysis.
    Higher values indicate better prospects for task completion.
    """
    import re
    
    # Convert to lowercase for easier matching
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Base Q-value starts at neutral
    q_value = 0.3
    
    # Check for explicit success in next state
    success_patterns = ['success', 'task completed', 'task completed!', 
                        'you have successfully', 'congratulations']
    for pattern in success_patterns:
        if pattern in next_state_lower:
            return 1.0
    
    # Check for explicit failure or error
    failure_patterns = ['error', 'invalid', 'cannot', 'nothing to', 'nothing at']
    for pattern in failure_patterns:
        if pattern in next_state_lower:
            return 0.0
    
    # Reward productive actions
    productive_action_patterns = [
        r'\bgo\s+to\b', r'\btake\s+\w+', r'\bput\s+\w+', r'\bclean\s+\w+',
        r'\bheat\s+\w+', r'\bcool\s+\w+', r'\bopen\s+\w+', r'\bclose\s+\w+',
        r'\btoggle\b', r'\bfind\b', r'\bput\s+in\b', r'\bput\s+on\b',
        r'\bput\s+inside\b', r'\bput\s+on\s+top\b'
    ]
    for pattern in productive_action_patterns:
        if re.search(pattern, action_lower):
            q_value += 0.15
            break
    
    # Check if we found a new object (progress indicator)
    object_keywords = [
        'apple', 'banana', 'plate', 'bowl', 'cup', 'mug', 'pan', 'pot',
        'tomato', 'egg', 'salt', 'pepper', 'book', 'pen', 'pencil', 'key',
        'remote', 'cellphone', 'laptop', 'cd', 'dvd', 'newspaper', 'magazine',
        'lettuce', 'potato', 'onion', 'garlic', 'bread', 'butter', 'milk',
        'juice', 'water', 'coffee', 'tea', 'wine', 'beer', 'soda', 'coke',
        'sprite', 'garbage', 'trash', 'sink', 'microwave', 'fridge', 'stove',
        'countertop', 'table', 'desk', 'drawer', 'cabinet', 'shelf', 'bed',
        'sofa', 'chair', 'dresser', 'dresser', 'nightstand', 'side table'
    ]
    
    state_objects = set()
    next_state_objects = set()
    for obj in object_keywords:
        if obj in state_lower:
            state_objects.add(obj)
        if obj in next_state_lower:
            next_state_objects.add(obj)
    
    # New objects found = progress
    new_objects = next_state_objects - state_objects
    if len(new_objects) > 0:
        q_value += 0.1
    
    # Check if we're holding an object (important progress state)
    holding_patterns = ['you are holding', 'you are carrying', 'holding', 'carrying']
    for pattern in holding_patterns:
        if pattern in next_state_lower:
            q_value += 0.1
            break
    
    # Check for location changes (navigation progress)
    if 'you are now' in next_state_lower or 'you arrive at' in next_state_lower:
        q_value += 0.05
    
    # Penalize if nothing changed (stuck state)
    state_tokens = set(state_lower.split())
    next_state_tokens = set(next_state_lower.split())
    overlap_ratio = len(state_tokens & next_state_tokens) / max(len(state_tokens), 1)
    if overlap_ratio > 0.9 and len(new_objects) == 0:
        q_value -= 0.1
    
    # Check for container interactions (often needed for task completion)
    container_keywords = ['microwave', 'fridge', 'sink', 'cabinet', 'drawer', 'shelf']
    for container in container_keywords:
        if container in next_state_lower:
            q_value += 0.05
            break
    
    # Penalize repeated action patterns (getting stuck)
    if action_lower in state_lower:
        q_value -= 0.05
    
    # Ensure value is in valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value