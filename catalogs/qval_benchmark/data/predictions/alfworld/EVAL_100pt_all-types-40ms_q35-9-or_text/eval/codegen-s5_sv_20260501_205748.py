def signal_function(state: str) -> float:
    import re
    
    # Check for terminal states
    state_lower = state.lower()
    
    # Complete success indicators
    if "completed" in state_lower or "success" in state_lower or "done" in state_lower or "goal reached" in state_lower:
        return 1.0
    
    # Failure indicators
    if "failed" in state_lower or "timeout" in state_lower or "limit reached" in state_lower or "episode ended" in state_lower:
        return 0.0
    
    # Extract step count if present
    remaining_steps = 40  # Default to full episode length
    
    # Pattern for "step X" or "step: X"
    step_match = re.search(r'step\s*(?::\s*)?(\d+)', state, re.IGNORECASE)
    if step_match:
        steps_used = int(step_match.group(1))
        remaining_steps = max(0, 40 - steps_used)
    
    # Pattern for "remaining" or "left"
    rem_match = re.search(r'(\d+)\s*(?:remaining|left|to go)', state, re.IGNORECASE)
    if rem_match:
        remaining_steps = int(rem_match.group(1))
    
    # Pattern for progress percentage
    progress_match = re.search(r'progress\s*(?::\s*)?([0-9.]+)%?', state, re.IGNORECASE)
    if progress_match:
        progress = float(progress_match.group(1))
        remaining_steps = max(0, int(40 * (1 - progress / 100)))
    
    # If no steps remaining and goal not reached, value is 0
    if remaining_steps <= 0:
        return 0.0
    
    # Calculate base value from remaining steps
    # Closer to goal = higher value (but not too close to limit)
    progress_ratio = 1 - (remaining_steps / 40)
    base_value = progress_ratio
    
    # Apply step-range adjustments
    if remaining_steps > 35:
        base_value *= 0.3  # Very early, high uncertainty
    elif remaining_steps > 25:
        base_value *= 0.5  # Early stage
    elif remaining_steps > 15:
        base_value *= 0.8  # Mid stage
    elif remaining_steps > 10:
        base_value *= 0.95  # Late stage, good progress
    else:
        base_value *= 0.98  # Very late, almost there
    
    # Check for positive goal-related keywords
    goal_keywords = ["pick", "drop", "move", "clean", "open", "close", "put", "take"]
    has_action = any(kw in state_lower for kw in goal_keywords)
    
    # Check for object state indicators
    state_keywords = ["dirty", "clean", "open", "closed", "charged", "empty", "full"]
    has_state = any(kw in state_lower for kw in state_keywords)
    
    # If we see action/state keywords, increase confidence
    if has_action or has_state:
        base_value *= 1.1
    
    # Clamp to valid range
    return max(0.0, min(1.0, base_value))