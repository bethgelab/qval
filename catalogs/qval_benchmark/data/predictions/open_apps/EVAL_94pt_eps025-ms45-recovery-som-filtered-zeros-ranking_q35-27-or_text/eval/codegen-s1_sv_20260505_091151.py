def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for task completion indicators
    completion_indicators = ['completed', 'success', 'done', 'saved', 'created', 'sent', 'added', 'event added', 'message sent', 'task completed']
    if any(ind in state_lower for ind in completion_indicators):
        return 1.0
    
    # Check for error or failure states
    error_indicators = ['error', 'failed', 'invalid', 'not found', 'cannot', 'unable']
    if any(ind in state_lower for ind in error_indicators):
        return 0.1
    
    # Score based on progress indicators
    score = 0.0
    
    # Check for form-related elements (indicates we're in the right place to complete task)
    form_indicators = ['input', 'textarea', 'form', 'field', 'name', 'date', 'time', 'subject', 'message', 'description', 'title']
    if any(ind in state_lower for ind in form_indicators):
        score += 0.3
    
    # Check for action buttons that could complete the task
    action_indicators = ['submit', 'save', 'send', 'add', 'create', 'button', 'confirm', 'apply', 'finish']
    if any(ind in state_lower for ind in action_indicators):
        score += 0.25
    
    # Check for task-specific content based on app type
    task_indicators = ['calendar', 'event', 'todo', 'message', 'chat', 'map', 'code', 'editor', 'schedule', 'reminder']
    if any(ind in state_lower for ind in task_indicators):
        score += 0.2
    
    # Check for bid tags (interactive elements available)
    import re
    bid_matches = re.findall(r'bid=\d+', state_lower)
    bid_count = len(bid_matches)
    if bid_count > 0:
        score += min(0.15, bid_count * 0.02)
    
    # Check for filled content (indicates progress made)
    filled_indicators = ['value=', 'filled', 'entered', 'typed', 'text=', 'content=']
    if any(ind in state_lower for ind in filled_indicators):
        score += 0.15
    
    # Check for navigation elements (indicates we can navigate if needed)
    nav_indicators = ['menu', 'link', 'tab', 'page', 'section', 'header', 'nav']
    if any(ind in state_lower for ind in nav_indicators):
        score += 0.1
    
    # Check for empty/blank state (low value)
    empty_indicators = ['empty', 'blank', 'no', 'none', 'nothing']
    if any(ind in state_lower for ind in empty_indicators):
        score = max(0.1, score * 0.7)
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, score))