def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Terminal success - highest possible value
    if any(term in state_lower for term in ['success', 'completed', 'task completed', 'you have successfully']):
        return 1.0
    
    # Terminal failure indicators
    if any(term in state_lower for term in ['fail', 'error', 'cannot find', 'nothing']):
        return 0.0
    
    # Base value starts at 0
    value = 0.0
    
    # Check if agent is holding an object (key progress indicator)
    if 'holding' in state_lower or 'in hand' in state_lower or 'you are holding' in state_lower:
        value += 0.35
    
    # Check for navigation progress (being in a room is better than not)
    if 'you are' in state_lower or 'in the' in state_lower:
        value += 0.15
    
    # Count task-relevant object mentions (more objects seen = better exploration)
    relevant_objects = ['apple', 'bowl', 'cup', 'plate', 'book', 'pen', 'potato', 'tomato', 
                       'egg', 'garlic', 'lettuce', 'microwave', 'fridge', 'sink', 'toaster',
                       'armchair', 'bed', 'desk', 'dresser', 'sidetable', 'sofa', 'table',
                       'cabinet', 'drawer', 'shelf', 'safe', 'tv', 'pc', 'alarmclock',
                       'bathroom', 'kitchen', 'bedroom', 'livingroom', 'garage', 'office']
    
    object_matches = sum(1 for obj in relevant_objects if obj in state_lower)
    value += min(object_matches * 0.03, 0.25)
    
    # Check for action keywords indicating task progress
    if 'pick' in state_lower or 'take' in state_lower:
        value += 0.1
    if 'put' in state_lower or 'place' in state_lower or 'receptacle' in state_lower:
        value += 0.1
    if 'go' in state_lower or 'move' in state_lower:
        value += 0.05
    
    # Bonus for specific high-value rooms
    if any(room in state_lower for room in ['kitchen', 'bedroom', 'livingroom']):
        value += 0.1
    
    # Penalize empty/uncertain states
    if 'empty' in state_lower or 'nothing' in state_lower:
        value -= 0.1
    
    # Ensure value is in valid range
    value = max(0.0, min(1.0, value))
    
    return value