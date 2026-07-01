import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for goal completion indicators
    completion_patterns = [
        r'success', r'completed', r'done', r'added', r'created',
        r'saved', r'sent', r'finished', r'confirmed', r'event.*created',
        r'message.*sent', r'task.*added', r'appointment.*scheduled',
        r'event.*saved', r'todo.*completed', r'checklist.*done'
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, state_lower):
            return 1.0
    
    # Base score starts at 0.5 for neutral state
    score = 0.5
    
    # Reward presence of task-relevant UI elements
    task_elements = ['input', 'form', 'button', 'submit', 'create', 'add', 'new', 'edit', 'save', 'send']
    element_count = sum(1 for elem in task_elements if elem in state_lower)
    score += min(0.25, element_count * 0.04)
    
    # Penalize error/problem states
    error_patterns = ['error', 'fail', 'invalid', 'required', 'missing', 'cannot', 'failed']
    error_count = sum(1 for err in error_patterns if err in state_lower)
    score -= error_count * 0.12
    
    # Reward presence of content suggesting we're on the right page
    content_indicators = ['calendar', 'todo', 'messenger', 'maps', 'editor', 'event', 'task', 'message', 'appointment', 'code']
    content_count = sum(1 for content in content_indicators if content in state_lower)
    score += min(0.20, content_count * 0.04)
    
    # Reward presence of bid tags (indicates interactive elements available)
    bid_pattern = r'bid\s*=\s*[\'\"]?\d+[\'\"]?'
    bid_matches = len(re.findall(bid_pattern, state_lower))
    score += min(0.15, bid_matches * 0.01)
    
    # Reward presence of navigation/interaction elements
    nav_elements = ['link', 'click', 'tab', 'menu', 'section', 'header', 'nav']
    nav_count = sum(1 for nav in nav_elements if nav in state_lower)
    score += min(0.10, nav_count * 0.025)
    
    # Penalize empty or minimal states
    if len(state_lower) < 100:
        score -= 0.15
    
    # Clamp to valid range
    score = max(0.0, min(1.0, score))
    
    return score