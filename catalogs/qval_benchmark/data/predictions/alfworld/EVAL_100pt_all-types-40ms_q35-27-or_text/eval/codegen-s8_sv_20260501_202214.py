def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check if task is completed
    if 'done' in state_lower or 'success' in state_lower or 'completed' in state_lower or 'goal' in state_lower and 'achieved' in state_lower:
        return 1.0
    
    # Check for failure indicators
    if 'fail' in state_lower or 'impossible' in state_lower or 'cannot' in state_lower:
        return 0.0
    
    # Base value
    value = 0.5
    
    # Positive indicators of progress
    if 'on' in state_lower and 'at' in state_lower:
        value += 0.15
    if 'holding' in state_lower or 'has' in state_lower:
        value += 0.1
    if 'near' in state_lower or 'close' in state_lower:
        value += 0.1
    if 'open' in state_lower:
        value += 0.05
    if 'clean' in state_lower:
        value += 0.05
    
    # Negative indicators
    if 'not' in state_lower:
        value -= 0.1
    if 'closed' in state_lower:
        value -= 0.05
    if 'dirty' in state_lower:
        value -= 0.05
    if 'empty' in state_lower:
        value -= 0.05
    
    # Check for location information (being in correct room helps)
    if 'in' in state_lower and ('room' in state_lower or 'kitchen' in state_lower or 'bedroom' in state_lower or 'bathroom' in state_lower):
        value += 0.1
    
    # Check for object presence
    if 'object' in state_lower or 'item' in state_lower:
        value += 0.05
    
    # Step efficiency bonus (if state indicates few steps remaining)
    if 'step' in state_lower:
        # If state mentions remaining steps, estimate based on that
        import re
        step_match = re.search(r'step.*?(\d+)', state_lower)
        if step_match:
            steps = int(step_match.group(1))
            if steps <= 10:
                value += 0.2
            elif steps <= 20:
                value += 0.1
    
    # Clamp value between 0 and 1
    return max(0.0, min(1.0, value))