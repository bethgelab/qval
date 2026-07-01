import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value estimate
    q_value = 0.0
    
    # Check if action is meaningful (not noop)
    action_lower = action.lower()
    is_noop = 'noop' in action_lower
    is_scroll = 'scroll' in action_lower
    
    # Action quality score
    if is_noop:
        action_quality = 0.0
    elif is_scroll:
        action_quality = 0.1
    elif 'click' in action_lower or 'fill' in action_lower or 'press' in action_lower:
        action_quality = 0.3
    else:
        action_quality = 0.1
    
    # Check for goal completion indicators in next_state
    success_indicators = [
        'success', 'completed', 'done', 'created', 'added', 'sent',
        'saved', 'finished', 'confirmed', 'event', 'message',
        'task', 'item', '✓', '✔', '✅'
    ]
    next_state_lower = next_state.lower()
    
    success_score = 0.0
    for indicator in success_indicators:
        if indicator in next_state_lower:
            success_score = max(success_score, 0.8)
    
    # Check for error/failure indicators
    error_indicators = ['error', 'failed', 'invalid', 'cannot', 'unable']
    for indicator in error_indicators:
        if indicator in next_state_lower:
            success_score = max(success_score, -0.3)
    
    # Check state change magnitude (meaningful progress)
    state_lower = state.lower()
    
    # Calculate simple similarity - if states are very similar, action had little effect
    state_len = len(state)
    next_len = len(next_state)
    if state_len == 0 or next_len == 0:
        state_change = 0.0
    else:
        # Count common words as a proxy for state change
        state_words = set(re.findall(r'\w+', state_lower))
        next_words = set(re.findall(r'\w+', next_state_lower))
        if len(state_words) == 0:
            state_change = 0.5
        else:
            common = len(state_words & next_words)
            union = len(state_words | next_words)
            similarity = common / union if union > 0 else 1.0
            state_change = 1.0 - similarity
    
    # State change bonus (but cap it to avoid overvaluing random changes)
    state_change_bonus = min(state_change * 0.4, 0.4)
    
    # Check for form-related elements (indicates we're in an interactive state)
    form_indicators = ['input', 'button', 'form', 'submit', 'label', 'field']
    form_score = 0.0
    for indicator in form_indicators:
        if indicator in next_state_lower:
            form_score = max(form_score, 0.2)
    
    # Check for navigation elements (links, menus)
    nav_indicators = ['link', 'menu', 'nav', 'href', 'tab']
    nav_score = 0.0
    for indicator in nav_indicators:
        if indicator in next_state_lower:
            nav_score = max(nav_score, 0.15)
    
    # Check if we're on a relevant page for common tasks
    task_pages = ['calendar', 'todo', 'messenger', 'maps', 'editor', 'message', 'event']
    page_score = 0.0
    for page in task_pages:
        if page in next_state_lower:
            page_score = max(page_score, 0.25)
            break
    
    # Bid element count (more interactive elements = more options)
    bid_matches = re.findall(r'bid\s*=\s*["\']?(\d+)["\']?', next_state)
    bid_count = len(bid_matches)
    bid_score = min(bid_count * 0.01, 0.3)
    
    # Combine scores
    q_value = (
        success_score * 0.5 +
        action_quality * 0.2 +
        state_change_bonus * 0.15 +
        form_score * 0.05 +
        nav_score * 0.05 +
        page_score * 0.03 +
        bid_score * 0.02
    )
    
    # Ensure Q-value is in reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)