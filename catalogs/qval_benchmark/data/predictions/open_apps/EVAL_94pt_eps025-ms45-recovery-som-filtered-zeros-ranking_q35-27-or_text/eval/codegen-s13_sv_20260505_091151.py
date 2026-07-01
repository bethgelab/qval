def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Strong success indicators - task likely complete
    strong_success = [
        'saved successfully', 'created successfully', 'sent successfully',
        'event created', 'task completed', 'message sent', 'item added',
        'success', 'completed', 'done', 'updated successfully'
    ]
    
    # Moderate success indicators
    moderate_success = [
        'saved', 'created', 'sent', 'added', 'updated',
        'calendar event', 'todo item', 'new message', 'event added'
    ]
    
    # Error/failure indicators
    error_patterns = [
        'error', 'failed', 'invalid', 'not found', 'failed to',
        'could not', 'unable', 'problem', 'issue'
    ]
    
    # App-specific content indicators
    app_content = [
        'calendar', 'event', 'todo', 'task', 'message', 'chat',
        'map', 'editor', 'code', 'search', 'result', 'list'
    ]
    
    # Interactive elements for progress
    interactive = [
        'button', 'submit', 'save', 'send', 'click', 'input',
        'form', 'field', 'text', 'link', 'bid='
    ]
    
    # Count matches
    strong_count = sum(1 for p in strong_success if p in state_lower)
    moderate_count = sum(1 for p in moderate_success if p in state_lower)
    error_count = sum(1 for p in error_patterns if p in state_lower)
    content_count = sum(1 for p in app_content if p in state_lower)
    interactive_count = sum(1 for p in interactive if p in state_lower)
    
    # Calculate value
    if strong_count > 0:
        # Task likely complete
        value = 0.9 + min(strong_count * 0.05, 0.1)
    elif error_count > 0:
        # Errors present, lower value
        value = 0.1 + max(0, 0.15 - error_count * 0.05)
    else:
        # Normal state - value based on progress indicators
        content_bonus = min(content_count * 0.05, 0.3)
        interactive_bonus = min(interactive_count * 0.03, 0.25)
        moderate_bonus = min(moderate_count * 0.1, 0.3)
        value = 0.3 + content_bonus + interactive_bonus + moderate_bonus
    
    return max(0.0, min(1.0, value))