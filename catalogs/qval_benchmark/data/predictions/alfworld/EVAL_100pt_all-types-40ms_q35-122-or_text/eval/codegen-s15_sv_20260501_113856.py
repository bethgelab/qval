def signal_function(state: str) -> float:
    """Estimate state-value for ALFWorld environment based on state text analysis."""
    
    import re
    
    state_lower = state.lower()
    
    # Terminal state checks - highest priority
    if any(term in state_lower for term in ['task completed', 'success', 'completed successfully']):
        return 1.0
    
    if any(term in state_lower for term in ['task failed', 'failed', 'timeout', 'step limit reached']):
        return 0.0
    
    # Initialize base value
    value = 0.0
    
    # Progress indicators - check if object has been found and picked up
    if 'holding' in state_lower:
        value += 0.35
    
    # Check if object has been placed in target location
    if 'put' in state_lower and ('on' in state_lower or 'in' in state_lower or 'at' in state_lower):
        value += 0.35
    
    # Navigation progress - being in relevant rooms
    if any(room in state_lower for room in ['kitchen', 'bedroom', 'living room', 'bathroom', 'garage', 'office', 'dining room']):
        value += 0.1
    
    # State-changing actions (cleaning, heating, cooling)
    if any(action in state_lower for action in ['clean', 'heat', 'cool', 'recharge', 'toggle']):
        value += 0.1
    
    # Object discovery progress
    if 'take' in state_lower or 'pick' in state_lower:
        value += 0.1
    
    # Penalize states near step limit (40 steps total)
    step_match = re.search(r'(\d+)\s*steps?\s*(?:remaining|of)', state_lower)
    if step_match:
        steps_remaining = int(step_match.group(1))
        if steps_remaining < 5:
            value *= 0.5  # Significant penalty for low steps
        elif steps_remaining < 10:
            value *= 0.8  # Moderate penalty
    
    # Clamp value to valid range
    value = min(1.0, max(0.0, value))
    
    return value