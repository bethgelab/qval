def signal_function(state: str) -> float:
    import math
    import re
    
    state_lower = state.lower()
    
    # Check if task is completed - return maximum value
    completion_indicators = ['success', 'completed', 'done', 'achieved', 'goal achieved', 'task complete', 'finished']
    for indicator in completion_indicators:
        if indicator in state_lower:
            return 1.0
    
    # Check for failure indicators
    failure_indicators = ['failed', 'cannot', 'impossible', 'error', 'timeout', 'limit reached']
    for indicator in failure_indicators:
        if indicator in state_lower:
            return 0.0
    
    # Extract goal-related information from state
    # Look for goal description patterns
    goal_pattern = re.search(r'(?:goal|task|objective|target)[:\s]+(.+?)(?:\n|$)', state_lower)
    goal_text = goal_pattern.group(1) if goal_pattern else ""
    
    # Count progress indicators in the state
    progress_score = 0.0
    total_possible = 0.0
    
    # Check if key objects mentioned in goal are in correct locations
    if goal_text:
        total_possible += 1.0
        # Look for object-location matches
        location_keywords = ['in the', 'on the', 'at the', 'near the', 'inside the']
        for loc_kw in location_keywords:
            if loc_kw in state_lower:
                # Check if goal objects appear with location keywords
                if any(word in goal_text for word in ['move', 'put', 'place', 'bring', 'take']):
                    progress_score += 0.3
                    break
    
    # Check agent position relative to task
    total_possible += 1.0
    agent_position_indicators = ['you are', 'your location', 'at ', 'in ']
    for indicator in agent_position_indicators:
        if indicator in state_lower:
            progress_score += 0.5
            break
    
    # Check for object states that indicate progress (opened, cleaned, etc.)
    total_possible += 1.0
    positive_states = ['opened', 'clean', 'turned on', 'heated', 'charged', 'refilled']
    for state_kw in positive_states:
        if state_kw in state_lower:
            progress_score += 0.25
            break
    
    # Check for containers being accessed (often needed for tasks)
    total_possible += 1.0
    container_actions = ['open', 'close', 'take', 'put', 'move', 'go to']
    for action in container_actions:
        if action in state_lower:
            progress_score += 0.2
            break
    
    # Normalize progress score
    if total_possible > 0:
        progress_ratio = min(1.0, progress_score / total_possible)
    else:
        progress_ratio = 0.5  # Default if no indicators found
    
    # Estimate remaining steps based on progress
    # Assume full task takes ~20 steps on average
    base_remaining_steps = 20
    estimated_remaining = base_remaining_steps * (1 - progress_ratio)
    
    # Penalize for being far from completion
    max_steps = 40
    if estimated_remaining >= max_steps:
        return 0.0
    
    # Calculate value using exponential decay
    # Closer to goal = higher value
    value = math.exp(-estimated_remaining / max_steps)
    
    # Boost value slightly for states showing clear progress
    if progress_ratio > 0.5:
        value = min(1.0, value * 1.2)
    
    return min(1.0, max(0.0, value))