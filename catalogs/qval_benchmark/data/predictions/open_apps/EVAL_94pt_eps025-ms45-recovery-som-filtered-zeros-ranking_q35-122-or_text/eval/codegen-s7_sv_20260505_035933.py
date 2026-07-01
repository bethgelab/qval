def signal_function(state: str) -> float:
    import re
    
    value = 0.0
    state_lower = state.lower()
    
    # Base value for being in a valid state
    value = 0.1
    
    # Check for app context - being in the right app is valuable
    app_indicators = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor']
    for app in app_indicators:
        if app in state_lower:
            value += 0.15
            break
    
    # Success/completion indicators - strong signal of goal achievement
    completion_words = [
        'success', 'completed', 'saved', 'sent', 'created', 'added', 
        'done', 'finished', 'confirmed', 'published', 'posted'
    ]
    completion_count = sum(1 for word in completion_words if word in state_lower)
    value += min(completion_count * 0.2, 0.6)
    
    # Count interactive elements (bids) - more options = more agency
    bid_matches = re.findall(r"bid['\"]?\s*[:=]\s*['\"]?(\d+)", state_lower)
    if len(bid_matches) > 0:
        value += min(len(bid_matches) * 0.015, 0.25)
    
    # Form/input elements suggest actionability toward goal
    form_words = ['input', 'form', 'text', 'field', 'button', 'submit', 'click', 'press', 'type']
    form_count = sum(1 for word in form_words if word in state_lower)
    value += min(form_count * 0.025, 0.2)
    
    # Navigation elements indicate ability to move toward goal
    nav_words = ['home', 'back', 'next', 'previous', 'menu', 'navigate', 'go', 'link', 'tab']
    nav_count = sum(1 for word in nav_words if word in state_lower)
    value += min(nav_count * 0.01, 0.1)
    
    # Task-specific progress indicators
    progress_words = [
        'event', 'meeting', 'appointment', 'task', 'note', 'message', 
        'todo', 'item', 'entry', 'record', 'location', 'destination'
    ]
    progress_count = sum(1 for word in progress_words if word in state_lower)
    value += min(progress_count * 0.03, 0.2)
    
    # Error indicators (negative signal)
    error_words = ['error', 'fail', 'invalid', 'missing', 'required', 'warning', 'not found', 'unable']
    error_count = sum(1 for word in error_words if word in state_lower)
    value -= min(error_count * 0.08, 0.3)
    
    # Content density - more meaningful content suggests progress
    line_count = len([l for l in state.split('\n') if l.strip()])
    if line_count > 15:
        value += 0.05
    if line_count > 40:
        value += 0.05
    
    # Check for specific action completions
    action_completions = [
        'event created', 'task added', 'message sent', 'todo completed',
        'note saved', 'appointment added', 'location found'
    ]
    action_count = sum(1 for action in action_completions if action in state_lower)
    value += min(action_count * 0.25, 0.5)
    
    # Penalize loading/empty states
    if 'loading' in state_lower or 'empty' in state_lower:
        value -= 0.1
    
    # Penalize modal/dialog blocking states (might need to close)
    if 'modal' in state_lower or 'dialog' in state_lower:
        value -= 0.05
    
    # Ensure value is in valid range
    value = max(0.0, min(1.0, value))
    
    return value