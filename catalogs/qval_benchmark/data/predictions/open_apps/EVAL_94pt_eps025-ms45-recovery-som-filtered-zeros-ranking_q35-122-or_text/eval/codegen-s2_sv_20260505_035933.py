def signal_function(state: str) -> float:
    import re
    
    # If task is complete, value is 1.0
    completion_patterns = [
        r'task.*complete', r'goal.*achieved', r'completed', r'success',
        r'saved', r'created', r'sent', r'added', r'confirmed', r'done'
    ]
    for pattern in completion_patterns:
        if re.search(pattern, state.lower()):
            return 1.0
    
    # Parse current step if available
    step_match = re.search(r'step[:\s]+(\d+)', state.lower())
    if step_match:
        current_step = int(step_match.group(1))
        remaining_steps = max(0, 45 - current_step)
    else:
        current_step = 0
        remaining_steps = 45
    
    # Time-based value: more remaining steps = higher chance to succeed
    time_value = min(1.0, remaining_steps / 45.0)
    
    # Count interactive elements (bids indicate actionable opportunities)
    bid_matches = re.findall(r"'bid':\s*'(\d+)'|bid['\"]?\s*[:=]\s*['\"]?(\d+)", state)
    bid_count = len([m for m in bid_matches if m[0] or m[1]])
    
    # More interactive elements = more opportunities to make progress
    action_value = min(1.0, bid_count / 10.0) * 0.3
    
    # Check for task progress indicators
    progress_indicators = [
        r'checked', r'ticked', r'task.*added', r'event.*added',
        r'message.*sent', r'route.*found', r'file.*saved',
        r'form.*filled', r'input.*entered'
    ]
    progress_count = sum(1 for p in progress_indicators if re.search(p, state.lower()))
    progress_value = min(1.0, progress_count / 3.0) * 0.4
    
    # Check for error/negative indicators (reduce value)
    error_patterns = [r'error', r'failed', r'invalid', r'not found', r'cannot']
    error_count = sum(1 for p in error_patterns if re.search(p, state.lower()))
    penalty = min(0.3, error_count * 0.1)
    
    # Combine factors: time + actions + progress - penalties
    estimated_value = (time_value * 0.3 + action_value + progress_value) - penalty
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, estimated_value))