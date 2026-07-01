def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for goal completion indicators - highest value
    completion_patterns = [
        'success', 'completed', 'done', 'created', 'added',
        'sent', 'saved', 'confirmed', 'finished', 'event created',
        'message sent', 'task added', 'appointment scheduled',
        'item created', 'entry added', 'form submitted'
    ]
    for pattern in completion_patterns:
        if pattern in state_lower:
            return 0.95
    
    # Check for error or failure indicators - lowest value
    error_patterns = [
        'error', 'failed', 'invalid', 'cannot', 'denied',
        'not found', 'permission denied', 'unauthorized',
        'access denied', 'timeout', 'connection failed'
    ]
    for pattern in error_patterns:
        if pattern in state_lower:
            return 0.05
    
    # Count progress indicators
    progress_patterns = [
        'form', 'input', 'field', 'create', 'add', 'new',
        'edit', 'update', 'send', 'save', 'submit', 'next',
        'continue', 'proceed', 'apply', 'confirm'
    ]
    progress_count = sum(1 for pattern in progress_patterns if pattern in state_lower)
    
    # Count interactive elements (bids indicate clickable/fillable elements)
    bid_matches = re.findall(r'bid=\d+', state)
    bid_count = len(bid_matches)
    
    # Count context indicators (being in the right app)
    context_patterns = [
        'todo', 'calendar', 'messenger', 'maps', 'editor',
        'event', 'task', 'message', 'location', 'code',
        'schedule', 'appointment', 'chat', 'conversation'
    ]
    context_count = sum(1 for pattern in context_patterns if pattern in state_lower)
    
    # Count filled content indicators (forms being populated)
    content_patterns = [
        'value=', 'placeholder', 'type=', 'required',
        'name=', 'id=', 'label', 'description'
    ]
    content_count = sum(1 for pattern in content_patterns if pattern in state_lower)
    
    # Calculate base value
    base_value = 0.25
    
    # Add value for progress indicators
    base_value += min(progress_count * 0.06, 0.30)
    
    # Add value for interactive elements (more options = more flexibility)
    base_value += min(bid_count * 0.008, 0.20)
    
    # Add value for being in relevant context
    base_value += min(context_count * 0.04, 0.25)
    
    # Add value for content being present (forms filled, etc.)
    base_value += min(content_count * 0.03, 0.15)
    
    # Check for navigation elements that suggest ability to move forward
    nav_patterns = ['link', 'button', 'click', 'navigate', 'go to']
    nav_count = sum(1 for pattern in nav_patterns if pattern in state_lower)
    base_value += min(nav_count * 0.03, 0.10)
    
    # Check for modal/dialog indicators (might need to close before proceeding)
    modal_patterns = ['modal', 'dialog', 'popup', 'overlay', 'confirm']
    modal_count = sum(1 for pattern in modal_patterns if pattern in state_lower)
    if modal_count > 0:
        base_value -= 0.05  # Slight penalty for needing to handle modals
    
    # Normalize to [0, 1]
    return max(0.0, min(1.0, base_value))