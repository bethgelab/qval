def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for error indicators - these significantly reduce value
    error_patterns = ['error', 'failed', 'invalid', 'cannot', 'unavailable', 'blocked', 'timeout', 'connection', 'exception']
    has_error = any(pattern in state_lower for pattern in error_patterns)
    
    # Check for goal-related positive indicators
    goal_keywords = ['calendar', 'event', 'message', 'send', 'todo', 'create', 'add', 'complete', 'success', 'done', 'task', 'appointment']
    has_goal_keywords = any(keyword in state_lower for keyword in goal_keywords)
    
    # Count bid elements (interactive elements) - more options = better
    bid_count = len(re.findall(r'bid\d+', state_lower))
    
    # Check for form-related indicators (task in progress)
    form_indicators = ['fill', 'input', 'text', 'checkbox', 'select', 'button', 'form']
    form_count = sum(state_lower.count(indicator) for indicator in form_indicators)
    
    # Check for navigation elements (ability to proceed)
    nav_indicators = ['link', 'menu', 'nav', 'header', 'footer', 'sidebar', 'toolbar']
    nav_count = sum(state_lower.count(indicator) for indicator in nav_indicators)
    
    # Check for page context (we're on a meaningful page)
    page_indicators = ['page', 'screen', 'view', 'dashboard', 'app', 'home', 'list', 'details', 'results', 'search']
    has_page_context = any(keyword in state_lower for keyword in page_indicators)
    
    # Check for completion indicators (very positive)
    completion_indicators = ['completed', 'finished', 'submitted', 'saved', 'created', 'added']
    has_completion = any(indicator in state_lower for indicator in completion_indicators)
    
    # Start with neutral value
    value = 0.5
    
    # Penalize errors heavily
    if has_error:
        value -= 0.4
    
    # Reward goal-related content
    if has_goal_keywords:
        value += 0.2
    
    # Reward having interactive elements to act on
    if bid_count > 0:
        value += min(bid_count * 0.02, 0.3)
    
    # Reward form elements (indicates task in progress)
    if form_count > 0:
        value += min(form_count * 0.02, 0.2)
    
    # Reward navigation options
    if nav_count > 0:
        value += min(nav_count * 0.02, 0.2)
    
    # Reward page context
    if has_page_context:
        value += 0.1
    
    # Reward completion indicators (strong signal of goal proximity)
    if has_completion:
        value += 0.2
    
    # Clamp to valid range [0.0, 1.0]
    return max(0.0, min(1.0, value))