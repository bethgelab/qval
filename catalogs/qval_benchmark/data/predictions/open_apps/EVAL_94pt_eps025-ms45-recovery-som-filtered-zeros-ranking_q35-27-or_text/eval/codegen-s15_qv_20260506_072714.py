import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Success indicators that suggest goal completion
    success_patterns = [
        r'completed', r'done', r'success', r'sent', r'added', r'created',
        r'saved', r'scheduled', r'posted', r'published', r'confirmed',
        r'✓', r'✔', r'checkmark', r'event.*created', r'task.*added',
        r'message.*sent', r'appointment.*saved'
    ]
    
    # Negative indicators suggesting failure or errors
    error_patterns = [
        r'error', r'failed', r'invalid', r'unable', r'not.*found',
        r'permission', r'denied', r'blocked', r'failed.*to'
    ]
    
    # Productive action patterns
    productive_actions = ['click', 'fill', 'press']
    wasteful_actions = ['noop']
    
    # Base score from action type
    base_score = 0.0
    if any(prod in action.lower() for prod in productive_actions):
        base_score = 0.3
    elif any(waste in action.lower() for waste in wasteful_actions):
        base_score = 0.05
    else:
        base_score = 0.15
    
    # Check for success indicators in next_state
    success_bonus = 0.0
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            success_bonus = max(success_bonus, 0.7)
    
    # Check for error indicators (reduce score)
    error_penalty = 0.0
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            error_penalty = max(error_penalty, 0.5)
    
    # Measure state progression (meaningful changes between state and next_state)
    state_length = len(state)
    next_state_length = len(next_state)
    length_change = abs(next_state_length - state_length)
    
    # Moderate state changes suggest productive interaction
    progression_bonus = 0.0
    if 50 < length_change < 500:
        progression_bonus = 0.2
    elif length_change > 500:
        progression_bonus = 0.15  # Large changes might be navigation
    elif length_change < 10:
        progression_bonus = 0.05  # Minimal change
    
    # Check if action contains specific element references (bid numbers)
    bid_pattern = re.search(r"bid['\"]?\s*=\s*['\"]?\d+", action)
    bid_bonus = 0.1 if bid_pattern else 0.0
    
    # Check for form-related actions (typically productive)
    form_keywords = ['input', 'text', 'field', 'form', 'submit', 'button']
    form_bonus = 0.0
    if any(keyword in action.lower() for keyword in form_keywords):
        form_bonus = 0.1
    
    # Check for navigation progress indicators
    nav_indicators = ['page', 'loaded', 'navigated', 'url', 'href']
    nav_bonus = 0.0
    if any(ind in next_state.lower() for ind in nav_indicators):
        nav_bonus = 0.05
    
    # Combine all components
    q_value = base_score + success_bonus + progression_bonus + bid_bonus + form_bonus + nav_bonus
    q_value = q_value - error_penalty
    
    # Clamp to valid Q-value range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    # Apply slight discount for actions that don't show clear progress
    if success_bonus < 0.1 and progression_bonus < 0.1 and q_value > 0.5:
        q_value *= 0.9
    
    return q_value