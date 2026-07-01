import re

def signal_function(state: str) -> float:
    # Check for terminal success state
    if re.search(r'success', state, re.IGNORECASE):
        return 1.0
    
    # Check for terminal failure state
    if re.search(r'failure|timeout|error|cannot', state, re.IGNORECASE):
        return 0.0
    
    # Initialize base value for unknown or distant states
    value = 0.1
    
    # Extract current room from observation
    room_match = re.search(r'You are (?:in|at) the? (\w+)', state, re.IGNORECASE)
    current_room = room_match.group(1).lower() if room_match else None
    
    # Extract target room from task description
    # Look for "to the [room]" or "in the [room]" in task/goal context
    task_match = re.search(r'(?:task|goal).*?to the? (\w+)', state, re.IGNORECASE)
    if not task_match:
        task_match = re.search(r'(?:task|goal).*?in the? (\w+)', state, re.IGNORECASE)
    target_room = task_match.group(1).lower() if task_match else None
    
    # Check if agent is holding the target object
    holding = bool(re.search(r'holding|in your hand', state, re.IGNORECASE))
    
    # Check location of objects mentioned in observation
    obj_loc_match = re.search(r'is (?:on|in) the? (\w+)', state, re.IGNORECASE)
    obj_location = obj_loc_match.group(1).lower() if obj_loc_match else None
    
    # Proximity to target room
    if current_room and target_room and current_room == target_room:
        value += 0.4
    elif current_room:
        value += 0.1
        
    # Inventory status bonus
    if holding:
        value += 0.3
        
    # Object location relative to target
    if obj_location and target_room and obj_location == target_room:
        value += 0.3
    elif obj_location:
        value += 0.1
        
    # High-value combinations (efficiency)
    if holding and current_room and target_room and current_room == target_room:
        value += 0.2  # Holding object and at target room (1 step to put)
        
    if obj_location and target_room and obj_location == target_room:
        value += 0.1  # Object already at target location
        
    # Clamp value between 0.0 and 1.0
    return min(1.0, max(0.0, value))