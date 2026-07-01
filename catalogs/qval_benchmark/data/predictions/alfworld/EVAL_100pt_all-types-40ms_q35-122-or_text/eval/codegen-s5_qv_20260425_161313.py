def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Check for terminal success in next_state
    success_patterns = [
        r'successfully', r'task completed', r'task finished', 
        r'congratulations', r'thank you', r'well done',
        r'you have successfully', r'task complete', r'completed'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state.lower()):
            return 1.0
    
    # Check for terminal failure in next_state
    failure_patterns = [
        r'failed', r'error', r'time out', r'limit reached', 
        r'exceeded', r'maximum steps'
    ]
    
    for pattern in failure_patterns:
        if re.search(pattern, next_state.lower()):
            return 0.0
    
    # Base Q-value for non-terminal states (discounted expected future reward)
    # Start with a moderate value assuming progress is being made
    q_value = 0.3
    
    # Analyze if action contributes to task progress
    action_lower = action.lower()
    
    # Navigation actions (necessary for reaching objects/locations)
    nav_actions = ['go to', 'navigate to', 'walk to', 'move to', 'enter', 'look']
    if any(nav in action_lower for nav in nav_actions):
        q_value += 0.1
    
    # Object interaction actions (direct progress toward goal)
    interact_actions = ['pick up', 'take', 'grab', 'put in', 'place in', 
                        'put on', 'place on', 'clean', 'heat', 'cool', 
                        'open', 'close', 'put', 'put']
    if any(act in action_lower for act in interact_actions):
        q_value += 0.15
    
    # Check if next_state indicates progress (more content, successful action)
    if len(next_state) > len(state) + 20:
        q_value += 0.05
    
    # Check for negative indicators in next_state
    negative_patterns = [
        r'cannot', r'nothing', r'not here', r'not found',
        r'nothing to', r'nothing in', r'no'
    ]
    for pattern in negative_patterns:
        if re.search(pattern, next_state.lower()):
            q_value -= 0.15
            break
    
    # Check for goal-relevant objects in next_state (indicates we're close)
    goal_objects = [
        'fridge', 'microwave', 'sink', 'table', 'counter', 'drawer',
        'cabinet', 'cup', 'plate', 'bowl', 'apple', 'tomato', 'egg',
        'potato', 'onion', 'garlic', 'salt', 'pepper', 'towel', 'soap',
        'lamp', 'tv', 'book', 'pen', 'paper', 'bottle', 'box', 'bag'
    ]
    
    object_count = sum(1 for obj in goal_objects if obj in next_state.lower())
    q_value += min(object_count * 0.03, 0.2)
    
    # Check for location keywords (being in the right place)
    locations = ['kitchen', 'bedroom', 'bathroom', 'living room', 'office',
                 'garage', 'basement', 'hallway', 'dining room']
    location_count = sum(1 for loc in locations if loc in next_state.lower())
    q_value += min(location_count * 0.05, 0.15)
    
    # Penalize if action seems to undo progress (e.g., putting down without purpose)
    if 'put down' in action_lower and 'pick up' not in action_lower:
        if 'inventory' not in next_state.lower() or 'nothing' in next_state.lower():
            q_value -= 0.1
    
    # Penalize if state didn't change meaningfully
    if abs(len(next_state) - len(state)) < 10 and 'cannot' not in next_state.lower():
        q_value -= 0.1
    
    # Ensure Q-value is in valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value