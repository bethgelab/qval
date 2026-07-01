def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for explicit success/completion indicators
    success_keywords = ['success', 'completed', 'task complete', 'accomplished', 'done']
    for keyword in success_keywords:
        if keyword in state_lower:
            return 1.0
    
    # Check for task accomplishment patterns (common in ALFWorld success states)
    accomplishment_patterns = [
        r'has been\s+(found|placed|moved|located)',
        r'is now\s+(in|on|at)\s+',
        r'you\s+(have|have\s+successfully)\s+(completed|done)',
        r'task\s+(complete|completed|done)',
    ]
    for pattern in accomplishment_patterns:
        if re.search(pattern, state_lower):
            return 1.0
    
    # Score based on progress indicators
    progress_score = 0.0
    
    # Task-related action indicators
    action_keywords = ['go', 'take', 'put', 'move', 'clean', 'wash', 'heat', 'cool', 'open', 'close', 'eat']
    for keyword in action_keywords:
        if keyword in state_lower:
            progress_score += 0.08
    
    # Object manipulation state indicators
    manipulation_indicators = ['holding', 'carrying', 'picked', 'taken', 'placed', 'put', 'located']
    for indicator in manipulation_indicators:
        if indicator in state_lower:
            progress_score += 0.12
    
    # Location-based progress (being at a location suggests navigation progress)
    location_indicators = ['in the', 'on the', 'at the', 'in ', 'on ', 'at ']
    for indicator in location_indicators:
        if indicator in state_lower:
            progress_score += 0.06
    
    # Check for inventory/possession (having needed objects is good progress)
    inventory_patterns = [
        r'you\s+(have|are\s+holding)',
        r'you\s+(are\s+carrying|are\s+holding)',
        r'now\s+you\s+(have|are)',
    ]
    for pattern in inventory_patterns:
        if re.search(pattern, state_lower):
            progress_score += 0.15
    
    # Check for object-state changes (washing, heating, cleaning = progress)
    state_change_indicators = ['clean', 'wash', 'heated', 'warmed', 'cooled', 'opened', 'closed']
    for indicator in state_change_indicators:
        if indicator in state_lower:
            progress_score += 0.1
    
    # Check for goal-related objects mentioned (suggests task is on track)
    goal_indicators = ['task', 'goal', 'object', 'item', 'place', 'location']
    for indicator in goal_indicators:
        if indicator in state_lower:
            progress_score += 0.05
    
    # Base value for being in a valid state (not stuck)
    base_value = 0.15
    
    # Add step efficiency bonus (if state suggests fewer steps needed)
    # Shorter responses often indicate simpler states closer to completion
    state_length = len(state_lower.split())
    if state_length < 30:
        efficiency_bonus = 0.15
    elif state_length < 50:
        efficiency_bonus = 0.10
    elif state_length < 80:
        efficiency_bonus = 0.05
    else:
        efficiency_bonus = 0.0
    
    # Calculate final estimate
    estimated_value = base_value + progress_score + efficiency_bonus
    
    # Ensure value is in valid range [0, 1)
    # Never return 1.0 unless explicitly detected as success
    return min(estimated_value, 0.99)