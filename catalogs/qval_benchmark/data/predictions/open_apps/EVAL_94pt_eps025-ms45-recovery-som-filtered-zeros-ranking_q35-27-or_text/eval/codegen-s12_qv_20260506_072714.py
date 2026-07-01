import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    # Goal achievement indicators - common success messages in OpenApps
    success_patterns = [
        r'created', r'added', r'sent', r'completed', r'updated',
        r'saved', r'success', r'done', r'finished', r'confirmed',
        r'event\s+created', r'task\s+added', r'message\s+sent',
        r'appointment\s+added', r'file\s+saved', r'code\s+executed'
    ]
    
    # Check if goal appears to be achieved in next_state
    next_state_lower = next_state.lower()
    goal_achieved = False
    for pattern in success_patterns:
        if re.search(pattern, next_state_lower):
            goal_achieved = True
            break
    
    # If goal achieved, Q-value is high (close to 1.0)
    if goal_achieved:
        return 0.95
    
    # Action type scoring - productive actions are better
    action_lower = action.lower()
    action_score = 0.0
    if 'noop' in action_lower:
        action_score = 0.1  # No-op is rarely productive
    elif 'click' in action_lower:
        action_score = 0.6  # Clicks are typically productive
    elif 'fill' in action_lower:
        action_score = 0.7  # Filling forms is very productive
    elif 'press' in action_lower:
        action_score = 0.55  # Key presses can be productive
    elif 'scroll' in action_lower:
        action_score = 0.2  # Scrolling is neutral
    else:
        action_score = 0.3  # Unknown action type
    
    # Measure state change - meaningful changes indicate progress
    # Compare key structural elements between states
    def extract_key_elements(s):
        # Extract common interactive element markers
        bids = re.findall(r"bid=['\"]?(\d+)['\"]?", s)
        # Extract text content length as proxy for content changes
        text_content = re.sub(r'<[^>]+>', ' ', s)
        return (len(bids), len(text_content), len(s))
    
    state_elements = extract_key_elements(state)
    next_elements = extract_key_elements(next_state)
    
    # Calculate change ratio
    state_len = state_elements[2]
    next_len = next_elements[2]
    
    if state_len > 0:
        change_ratio = abs(next_len - state_len) / state_len
    else:
        change_ratio = 0.0
    
    # Cap change ratio to avoid extreme values
    change_ratio = min(change_ratio, 1.0)
    
    # Check for form completion indicators
    form_patterns = [r'filled', r'input', r'textarea', r'submit', r'button']
    form_progress = sum(1 for p in form_patterns if re.search(p, next_state_lower))
    
    # Check if we're on a relevant page (not navigation error)
    error_patterns = [r'error', r'not\s+found', r'unavailable', r'failed']
    has_error = any(re.search(p, next_state_lower) for p in error_patterns)
    
    # Combine signals into Q-value estimate
    base_value = 0.0
    
    # Action productivity contribution
    base_value += action_score * 0.4
    
    # State change contribution (moderate changes are good)
    if 0.01 < change_ratio < 0.5:
        base_value += 0.2
    elif change_ratio > 0.5:
        base_value += 0.1  # Large changes might be navigation
    
    # Form progress contribution
    base_value += min(form_progress * 0.05, 0.15)
    
    # Penalty for errors
    if has_error:
        base_value -= 0.3
    
    # Ensure Q-value stays in valid range
    q_value = max(0.0, min(1.0, base_value))
    
    return q_value