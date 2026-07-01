def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check if task is already completed
    if any(term in state_lower for term in ['success', 'completed', 'done', 'finished', 'task completed']):
        return 1.0
    
    # Check for goal-related context
    goal_patterns = [
        r'goal.*place',
        r'put.*in',
        r'move.*to',
        r'bring.*to',
        r'clean.*in',
        r'cook.*on',
        r'wash.*in'
    ]
    has_goal = any(re.search(pattern, state_lower) for pattern in goal_patterns)
    
    # Check if agent has objects in inventory
    inventory_patterns = [
        r'you are holding',
        r'you have',
        r'inventory',
        r'with you',
        r'carrying'
    ]
    has_inventory = any(re.search(pattern, state_lower) for pattern in inventory_patterns)
    
    # Check if agent is at target location
    location_patterns = [
        r'you are in',
        r'you are at',
        r'on the',
        r'in the'
    ]
    has_location = any(re.search(pattern, state_lower) for pattern in location_patterns)
    
    # Check if target object is visible/available
    object_patterns = [
        r'there is',
        r'there are',
        r'you see',
        r'available'
    ]
    has_object = any(re.search(pattern, state_lower) for pattern in object_patterns)
    
    # Check if object needs state change (penalty)
    state_change_needed = any(term in state_lower for term in 
                              ['dirty', 'cold', 'raw', 'closed', 'un', 'not clean', 'not hot', 'not cooked'])
    
    # Check if agent is at wrong location (penalty)
    wrong_location = any(term in state_lower for term in 
                         ['not here', 'not there', 'far', 'far away', 'across'])
    
    # Calculate base value from positive indicators
    value = 0.0
    if has_goal:
        value += 0.25
    if has_inventory:
        value += 0.20
    if has_location:
        value += 0.15
    if has_object:
        value += 0.15
    
    # Apply penalties
    if state_change_needed:
        value -= 0.10
    if wrong_location:
        value -= 0.15
    
    # Check for intermediate progress markers
    progress_markers = ['found', 'located', 'ready', 'here', 'there', 'nearby', 'visible']
    if any(marker in state_lower for marker in progress_markers):
        value += 0.10
    
    # Clamp value to valid range
    value = max(0.0, min(1.0, value))
    
    return value