def signal_function(state: str) -> float:
    import re
    
    # Extract step number from state
    step_match = re.search(r'step[:\s]+(\d+)', state, re.IGNORECASE)
    current_step = int(step_match.group(1)) if step_match else 0
    
    total_steps = 45
    steps_remaining = max(0, total_steps - current_step)
    
    # Check for task completion indicators in state text
    completion_patterns = [
        r'success', r'completed', r'done', r'saved', r'sent',
        r'added', r'created', r'finished', r'confirmed', r'submitted',
        r'saved\s+successfully', r'task\s+completed', r'message\s+sent',
        r'event\s+added', r'appointment\s+scheduled', r'created\s+successfully'
    ]
    
    completion_count = sum(1 for p in completion_patterns if re.search(p, state, re.IGNORECASE))
    
    # Count interactive elements (bids) to gauge remaining complexity
    bid_count = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?(\d+)", state))
    
    # Count form fields that may need to be filled
    form_fields = len(re.findall(r"(input|textarea|select)", state, re.IGNORECASE))
    
    # Base value from remaining steps (more steps = more opportunity to complete)
    step_value = steps_remaining / total_steps if total_steps > 0 else 0.0
    
    # Completion bonus - strong signal of progress toward goal
    completion_bonus = min(1.0, completion_count * 0.3)
    
    # Discount factor - earlier completion is more valuable
    discount = 0.95 ** current_step
    
    # Complexity penalty - more form fields and bids may indicate more work remaining
    complexity_penalty = min(1.0, form_fields * 0.05 + bid_count * 0.02)
    
    # Combined value estimate
    value = (step_value * 0.3 + completion_bonus * 0.7) * discount * (1.0 - complexity_penalty * 0.5)
    
    # Boost value if strong completion signals detected
    if completion_count >= 3:
        value = max(value, 0.85)
    elif completion_count >= 2:
        value = max(value, 0.65)
    elif completion_count >= 1:
        value = max(value, 0.45)
    
    # Ensure value is in valid range [0, 1]
    return min(1.0, max(0.0, value))