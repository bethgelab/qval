def signal_function(state: str) -> float:
    import re
    
    # Check for explicit completion indicators
    completion_patterns = [
        r'(?i)task\s*(completed|done|finished|success)',
        r'(?i)goal\s*(reached|achieved|completed)',
        r'(?i)(event|appointment|todo|message)\s*(created|added|sent|saved)',
        r'(?i)success\s*(message|notification)?',
        r'(?i)confirmation',
        r'(?i)saved\s+successfully',
        r'(?i)sent\s+successfully',
        r'(?i)created\s+successfully',
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, state):
            return 1.0
    
    # Extract step count if available in state
    step_match = re.search(r'(?:step|step\s*count|steps?\s*:?\s*)(\d+)', state, re.IGNORECASE)
    current_step = int(step_match.group(1)) if step_match else 0
    
    # Extract max steps if available (default to 45)
    max_step_match = re.search(r'(?:max|limit|total)\s*(?:step|steps)\s*:?\s*(\d+)', state, re.IGNORECASE)
    max_steps = int(max_step_match.group(1)) if max_step_match else 45
    
    # Calculate step-based progress value
    if max_steps > 0 and current_step < max_steps:
        steps_remaining = max_steps - current_step
        step_ratio = steps_remaining / max_steps
    else:
        step_ratio = 0.5
    
    # Count interactive elements (bids indicate available actions)
    bid_matches = re.findall(r'bid["\s=:]+["\']?(\d+)["\']?', state)
    bid_count = len(bid_matches)
    
    # Count filled form fields or entered text
    filled_patterns = [
        r'value["\s=:]+["\']([^"\']+)',
        r'filled["\s=:]+["\']([^"\']+)',
        r'text["\s=:]+["\']([^"\']+)',
        r'content["\s=:]+["\']([^"\']+)',
    ]
    filled_count = 0
    for pattern in filled_patterns:
        filled_count += len(re.findall(pattern, state))
    
    # Count navigation elements (links, buttons)
    link_count = len(re.findall(r'(?i)(?:link|button|clickable|tab|menu)', state))
    
    # Base value estimation
    base_value = 0.0
    
    # Reward for having many action options (exploration potential)
    if bid_count > 15:
        base_value += 0.15
    elif bid_count > 8:
        base_value += 0.10
    elif bid_count > 3:
        base_value += 0.05
    
    # Reward for making progress (filling forms, entering data)
    if filled_count > 0:
        base_value += 0.08 * min(filled_count, 4)
    
    # Reward for navigation capability
    if link_count > 5:
        base_value += 0.08
    elif link_count > 2:
        base_value += 0.04
    
    # Time pressure penalty (less value if running out of steps)
    if step_ratio < 0.3:
        base_value *= 0.6
    elif step_ratio < 0.5:
        base_value *= 0.8
    
    # Combine step ratio with action potential
    value = base_value * step_ratio
    
    # Bonus for recent progress (if filled_count increased relative to step count)
    if current_step > 0 and filled_count > 0:
        progress_rate = filled_count / current_step
        if progress_rate > 0.2:
            value += 0.05
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, value))