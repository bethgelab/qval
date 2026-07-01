def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for task completion
    if "success" in state_lower or "completed" in state_lower or "goal achieved" in state_lower:
        return 1.0
    
    # Check for failure/stuck states
    if "error" in state_lower or "invalid" in state_lower or "cannot" in state_lower:
        return 0.0
    
    # Base value
    value = 0.0
    
    # Progress indicators
    progress_score = 0.0
    
    # Holding objects indicates progress on many tasks
    if "holding" in state_lower:
        progress_score += 0.25
    
    # Valid actions available (not stuck)
    if "you can" in state_lower or "available" in state_lower:
        progress_score += 0.15
    
    # At correct location (common goal locations)
    locations = ["kitchen", "bedroom", "bathroom", "living room", "dining room"]
    for loc in locations:
        if loc in state_lower:
            progress_score += 0.1
            break
    
    # Object in container/on surface (good for many tasks)
    if "in" in state_lower and ("container" in state_lower or "box" in state_lower or "cupboard" in state_lower or "drawer" in state_lower or "sink" in state_lower):
        progress_score += 0.15
    
    # Step information if available - check if we're early in episode
    if "step" in state_lower:
        try:
            step_match = re.search(r'\d+/\d+', state_lower)
            if step_match:
                step_info = step_match.group()
                parts = step_info.split('/')
                if len(parts) == 2:
                    current = int(parts[0])
                    total = int(parts[1])
                    if current < total * 0.5:
                        progress_score += 0.1
        except:
            pass
    
    # Calculate final value
    value = min(1.0, progress_score)
    
    return value