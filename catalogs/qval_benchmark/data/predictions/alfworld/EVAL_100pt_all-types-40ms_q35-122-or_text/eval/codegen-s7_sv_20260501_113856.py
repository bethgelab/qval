def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for explicit success patterns
    success_patterns = [
        'task completed',
        'you have successfully',
        'congratulations',
        'task accomplished',
        'goal achieved',
        'all done'
    ]
    
    for pattern in success_patterns:
        if pattern in state_lower:
            return 1.0
    
    # Check for failure/terminal patterns
    failure_patterns = [
        'step limit reached',
        'maximum steps',
        'you have exceeded'
    ]
    
    for pattern in failure_patterns:
        if pattern in state_lower:
            return 0.0
    
    # Estimate task progress based on observation patterns
    progress_score = 0.0
    progress_count = 0
    
    # Object acquisition (picking up)
    if 'you pick up' in state_lower or 'you take' in state_lower:
        progress_score += 0.2
        progress_count += 1
    
    # Object placement (putting in/on)
    if 'put' in state_lower and ('in' in state_lower or 'on' in state_lower):
        progress_score += 0.3
        progress_count += 1
    
    # Navigation progress (going to rooms)
    if 'go to' in state_lower or 'you go to' in state_lower:
        progress_score += 0.1
        progress_count += 1
    
    # State modification tasks (heat/cool/clean)
    if 'heat' in state_lower or 'cool' in state_lower or 'clean' in state_lower:
        progress_score += 0.2
        progress_count += 1
    
    # Target location indicators
    target_locations = ['table', 'counter', 'sink', 'microwave', 'fridge', 'drawer', 'shelf', 'bed', 'sofa']
    for loc in target_locations:
        if loc in state_lower:
            progress_score += 0.1
            progress_count += 1
            break
    
    # Container interactions
    if 'drawer' in state_lower or 'cabinet' in state_lower or 'box' in state_lower:
        progress_score += 0.1
        progress_count += 1
    
    # Calculate base progress estimate
    if progress_count > 0:
        base_progress = progress_score / min(progress_count, 5)
    else:
        base_progress = 0.1
    
    # Boost value if multiple task components are visible
    if progress_count >= 3:
        base_progress = min(base_progress + 0.2, 0.8)
    elif progress_count >= 2:
        base_progress = min(base_progress + 0.15, 0.7)
    elif progress_count >= 1:
        base_progress = min(base_progress + 0.1, 0.6)
    
    # Check for proximity keywords
    if 'near' in state_lower or 'close' in state_lower or 'next to' in state_lower:
        base_progress = min(base_progress + 0.15, 0.9)
    
    # Check if agent has the object (being carried)
    if 'holding' in state_lower or 'carrying' in state_lower:
        base_progress = min(base_progress + 0.15, 0.85)
    
    # Ensure value is in valid range
    return max(0.0, min(1.0, base_progress))