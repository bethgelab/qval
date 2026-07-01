def signal_function(state: str) -> float:
    import re
    import math
    
    # Base value for neutral state
    value = 0.0
    
    # Check for goal-related success indicators
    success_patterns = [
        r'success',
        r'complete',
        r'done',
        r'achieved',
        r'verified',
        r'task.*complete',
        r'goal.*reach',
    ]
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            value = max(value, 0.95)
            break
    
    # Check for failure/error indicators
    failure_patterns = [
        r'error',
        r'fail',
        r'invalid',
        r'missing',
        r'not.*found',
        r'cannot',
    ]
    for pattern in failure_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            value = max(0.0, value - 0.3)
            break
    
    # Check for progress indicators
    progress_patterns = [
        r'progress',
        r'complete',
        r'percent',
        r'step.*current',
        r'of.*total',
        r'remaining',
    ]
    for pattern in progress_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            value = max(value + 0.1, value)
            break
    
    # Check for form elements (good for tasks requiring data entry)
    form_patterns = [
        r'input',
        r'textarea',
        r'button',
        r'submit',
        r'form',
        r'checkbox',
        r'radio',
    ]
    form_count = len(re.findall(r'\b(input|textarea|button|form|checkbox|radio)\b', state, re.IGNORECASE))
    if form_count > 0:
        value = min(value + 0.15 * form_count, value + 0.6)
    
    # Check for bid numbers (indicates interactive elements available)
    bid_matches = re.findall(r'\bid\d+', state)
    if bid_matches:
        num_bids = len(bid_matches)
        if num_bids >= 3:
            value = min(value + 0.05 * (num_bids - 2), value + 0.5)
    
    # Check for page context indicators
    page_context = [
        r'calendar',
        r'message',
        r'messenger',
        r'todo',
        r'code',
        r'editor',
        r'map',
        r'navigation',
    ]
    context_found = any(re.search(p, state, re.IGNORECASE) for p in page_context)
    if context_found:
        value = min(value + 0.1, value + 0.4)
    
    # Check for goal-specific task language
    task_keywords = [
        r'add.*event',
        r'send.*message',
        r'create.*todo',
        r'edit',
        r'update',
        r'new',
    ]
    for keyword in task_keywords:
        if re.search(keyword, state, re.IGNORECASE):
            value = min(value + 0.05, value + 0.5)
            break
    
    # Check for scroll position (indicates page navigation progress)
    scroll_patterns = [
        r'scroll',
        r'viewport',
        r'position',
        r'offset',
    ]
    for pattern in scroll_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            value = min(value + 0.02, value + 0.3)
            break
    
    # Clamp value to reasonable range [0, 1]
    value = max(0.0, min(1.0, value))
    
    return float(value)