def signal_function(state: str) -> float:
    """
    Estimate state-value for ALFWorld environment.
    Returns a value between 0 and 1 based on task progress indicators.
    """
    import re
    
    state_lower = state.lower()
    
    # Terminal states
    if any(term in state_lower for term in ['task completed', 'completed the task', 'success', 'you have completed']):
        return 1.0
    
    if any(term in state_lower for term in ['game over', 'failed', 'exceeded step limit', 'maximum steps']):
        return 0.0
    
    # Extract current room
    current_room = None
    room_match = re.search(r'you are in the (\w+)', state_lower)
    if room_match:
        current_room = room_match.group(1)
    
    # Extract inventory
    inventory = []
    inv_match = re.search(r'you are carrying: (.*?)(?:\n|$)', state_lower)
    if inv_match:
        inv_text = inv_match.group(1)
        if 'nothing' not in inv_text:
            inventory = [obj.strip() for obj in inv_text.split(',')]
    
    # Extract visible objects
    visible = []
    vis_match = re.search(r'you see: (.*?)(?:\n|$)', state_lower)
    if vis_match:
        vis_text = vis_match.group(1)
        if 'nothing' not in vis_text:
            visible = [obj.strip() for obj in vis_text.split(',')]
    
    # Estimate progress based on key indicators
    progress = 0.0
    
    # Base value for being in a valid state
    progress += 0.1
    
    # Check if agent is carrying something (found an object)
    if len(inventory) > 0:
        progress += 0.2
    
    # Check for completion indicators in state text
    completion_phrases = ['put', 'placed', 'on the', 'in the', 'to the']
    completion_score = sum(1 for phrase in completion_phrases if phrase in state_lower)
    progress += min(completion_score * 0.1, 0.3)
    
    # Penalize being in wrong room (not kitchen, living room, bedroom, bathroom, office)
    valid_rooms = ['kitchen', 'livingroom', 'bedroom', 'bathroom', 'office', 'garage', 'diningroom']
    if current_room and current_room not in valid_rooms:
        progress -= 0.1
    
    # Penalize if nothing is visible (agent may be stuck)
    if len(visible) == 0:
        progress -= 0.1
    
    # Ensure value is in valid range
    progress = max(0.0, min(1.0, progress))
    
    return progress