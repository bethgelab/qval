def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Start with base Q-value
    q_value = 0.0
    
    # Check for immediate success
    success_indicators = ['task completed', 'congratulations', 'you have', 'solved', 'success']
    next_lower = next_state.lower()
    for indicator in success_indicators:
        if indicator in next_lower:
            return 1.0
    
    # Check if action was invalid or led to error
    error_indicators = ['nothing to take', 'nothing to put', 'cannot', 'cannot find', 'invalid', 'error', 'no such']
    action_lower = action.lower()
    state_lower = state.lower()
    for indicator in error_indicators:
        if indicator in next_lower or indicator in action_lower:
            return -0.2
    
    # Identify productive action types
    productive_actions = ['go to', 'go to the', 'look at', 'take', 'put', 'clean', 'heat', 'cool', 
                          'open', 'close', 'on', 'in', 'inside', 'with', 'from']
    
    is_productive = any(prod in action_lower for prod in productive_actions)
    
    # Check for wasteful actions
    wasteful_patterns = ['go back', 'back', 'again', 're', 'repeat']
    is_wasteful = any(wast in action_lower for wast in wasteful_patterns)
    
    # Extract goal-related objects from state
    goal_objects = []
    if 'put' in state_lower:
        put_match = re.search(r'put\s+(\w+)\s+(?:in|on|inside)\s+(\w+)', state_lower)
        if put_match:
            goal_objects.append(put_match.group(1))
            goal_objects.append(put_match.group(2))
    
    # Check if action involves goal objects
    action_involves_goal = any(obj in action_lower for obj in goal_objects)
    
    # Check if action involves any object (object manipulation)
    object_keywords = ['apple', 'banana', 'book', 'bowl', 'box', 'cup', 'pen', 'plate', 'potato', 
                       'tomato', 'vase', 'watch', 'key', 'cellphone', 'creditcard', 'desklamp',
                       'soapbar', 'soapbottle', 'towel', 'cloth', 'spraybottle', 'statue', 
                       'sinkbasin', 'countertop', 'drawer', 'cabinet', 'dresser', 'garbagecan',
                       'microwave', 'fridge', 'stove', 'sink', 'coffeemachine', 'toaster',
                       'sofa', 'sidetable', 'shelf', 'table', 'bed', 'armchair', 'chair',
                       'drawer', 'cabinet', 'cupboard', 'cooler', 'pan', 'pot', 'kettle']
    
    involves_object = any(obj in action_lower for obj in object_keywords)
    
    # Calculate Q-value based on action quality
    if is_productive and not is_wasteful:
        q_value = 0.3
        if action_involves_goal:
            q_value += 0.3
        if involves_object:
            q_value += 0.1
    elif is_wasteful:
        q_value = -0.1
    else:
        q_value = 0.0
    
    # Bonus for actions that change state meaningfully
    if len(next_state) > len(state) * 0.9 and len(next_state) < len(state) * 1.5:
        q_value += 0.05
    
    # Ensure value is in reasonable range
    q_value = max(-0.5, min(1.0, q_value))
    
    return q_value