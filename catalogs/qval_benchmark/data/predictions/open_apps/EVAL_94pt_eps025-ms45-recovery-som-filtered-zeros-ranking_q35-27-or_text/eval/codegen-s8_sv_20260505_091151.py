def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for goal achievement indicators
    success_patterns = [
        'success', 'completed', 'done', 'finished', 'confirmed',
        'added', 'created', 'sent', 'saved', 'updated',
        'event created', 'message sent', 'task completed',
        '✓', '✔', '✅', 'check', 'verified', 'accepted'
    ]
    
    for pattern in success_patterns:
        if pattern in state_lower:
            return 0.95
    
    # Check for failure/error indicators
    failure_patterns = [
        'error', 'failed', 'invalid', 'denied', 'blocked',
        'not found', 'unauthorized', 'timeout', 'exception'
    ]
    
    for pattern in failure_patterns:
        if pattern in state_lower:
            return 0.15
    
    # Base score starts moderate
    base_score = 0.4
    
    # Count bid tags (interactive elements) - more options = better
    bid_matches = re.findall(r"bid['\"]?\s*[:=]\s*['\"]?\d+['\"]?", state)
    bid_count = len(bid_matches)
    if bid_count > 0:
        base_score += min(0.25, bid_count * 0.03)
    
    # Check for form/task-relevant elements
    form_elements = ['input', 'button', 'submit', 'form', 'field', 'text', 'textarea']
    form_count = sum(1 for elem in form_elements if elem in state_lower)
    base_score += min(0.15, form_count * 0.04)
    
    # Check for navigation elements
    nav_elements = ['link', 'click', 'navigate', 'go to', 'menu', 'tab']
    nav_count = sum(1 for elem in nav_elements if elem in state_lower)
    base_score += min(0.1, nav_count * 0.03)
    
    # Check for task-specific content (calendar, message, todo, etc.)
    content_patterns = ['calendar', 'event', 'message', 'todo', 'task', 
                       'date', 'time', 'recipient', 'subject', 'title',
                       'description', 'map', 'location', 'code', 'editor']
    content_count = sum(1 for pattern in content_patterns if pattern in state_lower)
    base_score += min(0.2, content_count * 0.04)
    
    # Penalize empty or sparse states
    if len(state) < 100:
        base_score *= 0.7
    
    # Penalize states with no clear interactive elements
    if bid_count == 0 and form_count < 2:
        base_score *= 0.8
    
    # Cap the score
    return min(0.85, max(0.05, base_score))