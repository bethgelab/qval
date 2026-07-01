def signal_function(state: str) -> float:
    import re
    
    # Convert state to lowercase for easier parsing
    state_lower = state.lower()
    
    # Check for task completion
    completion_indicators = [
        "your task is completed",
        "task completed",
        "completed",
        "success",
        "you have successfully"
    ]
    
    for indicator in completion_indicators:
        if indicator in state_lower:
            return 1.0
    
    # Extract task description to understand what object we need
    # Common task patterns in ALFWorld
    task_patterns = [
        r"put\s+(?:a\s+)?(\w+)\s+in/on\s+(?:a\s+)?(\w+)",
        r"find\s+(?:a\s+)?(\w+)\s+and\s+put\s+(?:it\s+)?in/on\s+(?:a\s+)?(\w+)",
        r"clean\s+(?:a\s+)?(\w+)\s+with\s+(?:a\s+)?(\w+)",
        r"heat\s+(?:a\s+)?(\w+)\s+with\s+(?:a\s+)?(\w+)",
        r"cool\s+(?:a\s+)?(\w+)\s+with\s+(?:a\s+)?(\w+)"
    ]
    
    target_object = None
    target_location = None
    
    for pattern in task_patterns:
        match = re.search(pattern, state_lower)
        if match:
            target_object = match.group(1)
            target_location = match.group(2)
            break
    
    # Check if target object is in inventory
    inventory_indicators = [
        "you are carrying",
        "inventory",
        "you have"
    ]
    
    has_target_object = False
    for indicator in inventory_indicators:
        if indicator in state_lower and target_object:
            # Check if the target object is mentioned near inventory indicators
            if target_object in state_lower:
                # Simple heuristic: if target object mentioned, assume we have it
                has_target_object = True
                break
    
    # Check if we're at the target location
    location_indicators = [
        "you are in the",
        "you arrive at the",
        "you enter the"
    ]
    
    at_target_location = False
    if target_location:
        for indicator in location_indicators:
            if indicator in state_lower:
                # Check if target location mentioned
                if target_location in state_lower:
                    at_target_location = True
                    break
    
    # Check for objects in current room
    room_has_target = False
    if target_object:
        # Look for target object mentioned in room description
        room_patterns = [
            r"on\s+(?:the\s+)?(\w+)",
            r"(\w+)\s+on\s+(?:the\s+)?(\w+)",
            r"(\w+)\s+in\s+(?:the\s+)?(\w+)"
        ]
        for pattern in room_patterns:
            matches = re.findall(pattern, state_lower)
            for match in matches:
                if target_object in match[0] or target_object in match[1]:
                    room_has_target = True
                    break
    
    # Estimate value based on progress
    # Start with base value
    value = 0.1
    
    # Bonus for having the target object
    if has_target_object:
        value += 0.5
    
    # Bonus for being at target location
    if at_target_location:
        value += 0.2
    
    # Bonus for object being in room
    if room_has_target:
        value += 0.15
    
    # Penalize for being stuck (repeated actions)
    # Check for repeated action patterns
    action_patterns = [
        r"go to",
        r"take",
        r"put",
        r"clean",
        r"heat",
        r"cool"
    ]
    
    # Count how many different action types mentioned
    action_count = sum(1 for pattern in action_patterns if pattern in state_lower)
    
    # If very few actions taken, lower value (early game)
    if action_count < 2:
        value -= 0.1
    
    # Ensure value is in valid range
    value = max(0.0, min(1.0, value))
    
    return float(value)