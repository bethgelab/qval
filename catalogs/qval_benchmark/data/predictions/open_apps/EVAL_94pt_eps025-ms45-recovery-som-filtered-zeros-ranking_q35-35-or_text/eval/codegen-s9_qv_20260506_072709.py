def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value for neutral state
    q_value = 0.5
    
    # Features that indicate goal proximity
    goal_indicators = [
        'event created', 'event saved', 'calendar event',
        'message sent', 'message delivered', 'chat sent',
        'task added', 'task created', 'todo added',
        'page saved', 'code saved', 'file saved',
        'location added', 'route added', 'marker added',
        'form submitted', 'form completed', 'form filled',
        'success', 'completed', 'done', 'finished'
    ]
    
    # Features that indicate progress
    progress_indicators = [
        'filling', 'entering', 'typing', 'navigating',
        'loading', 'processing', 'confirming', 'submitting',
        'clicking', 'selecting', 'choosing', 'opening',
        'editing', 'updating', 'creating', 'adding'
    ]
    
    # Features that indicate errors or dead ends
    error_indicators = [
        'error', 'failed', 'invalid', 'unavailable',
        'not found', 'cannot', 'unable', 'disabled',
        'locked', 'blocked', 'denied'
    ]
    
    # Check if goal indicators present in next_state (high reward signal)
    next_state_lower = next_state.lower()
    goal_matches = sum(1 for indicator in goal_indicators if indicator in next_state_lower)
    if goal_matches > 0:
        q_value = 1.0
    
    # Check for progress indicators in action or state
    action_lower = action.lower()
    state_lower = state.lower()
    progress_matches = sum(1 for indicator in progress_indicators if indicator in action_lower or indicator in state_lower)
    if progress_matches > 0:
        q_value = max(q_value, 0.7)
    
    # Check for error indicators (reduce Q-value)
    error_matches = sum(1 for indicator in error_indicators if indicator in next_state_lower)
    if error_matches > 0:
        q_value = max(0.0, q_value - 0.3)
    
    # Action type heuristics
    action_type = 'noop'
    if action.startswith('click'):
        action_type = 'click'
        q_value = max(q_value, 0.5)  # Clicking is usually productive
    elif action.startswith('fill'):
        action_type = 'fill'
        q_value = max(q_value, 0.6)  # Filling forms is progress
    elif action.startswith('press'):
        action_type = 'press'
        q_value = max(q_value, 0.5)  # Pressing keys can be progress
    elif action.startswith('scroll'):
        action_type = 'scroll'
        q_value = max(q_value, 0.4)  # Scrolling is preparatory
    elif action.startswith('noop'):
        action_type = 'noop'
        q_value = max(q_value, 0.3)  # Noop is usually not progress
    
    # State feature analysis - check for common app elements
    app_elements = {
        'todo': ['task', 'todo', 'add task', 'new task', 'checkbox'],
        'calendar': ['event', 'calendar', 'date', 'time', 'meeting'],
        'messenger': ['message', 'chat', 'send', 'recipient', 'inbox'],
        'maps': ['location', 'address', 'map', 'route', 'marker'],
        'code': ['code', 'file', 'editor', 'save', 'syntax']
    }
    
    state_elements_found = []
    for app_name, elements in app_elements.items():
        for element in elements:
            if element.lower() in state_lower:
                state_elements_found.append(app_name)
    
    # If we see app-specific elements, slightly boost Q-value
    if len(state_elements_found) > 0:
        q_value = max(q_value, 0.55)
    
    # Compare state to next_state - if elements were added, that's progress
    state_words = set(state_lower.split())
    next_state_words = set(next_state_lower.split())
    new_elements = next_state_words - state_words
    
    # Filter meaningful new elements (not just common words)
    meaningful_new = [w for w in new_elements if len(w) > 3 and w not in 
                     ['the', 'and', 'for', 'are', 'was', 'had', 'has', 'with', 'this', 'that']]
    
    if len(meaningful_new) > 0:
        q_value = max(q_value, 0.6)
    
    # Check if action matches state context (context-aware heuristic)
    context_matches = 0
    if 'button' in state_lower and action.startswith('click'):
        context_matches += 1
    if ('form' in state_lower or 'input' in state_lower) and action.startswith('fill'):
        context_matches += 1
    if ('link' in state_lower or 'navigate' in state_lower) and action.startswith('click'):
        context_matches += 1
    
    if context_matches > 0:
        q_value = max(q_value, 0.65)
    
    # Clamp Q-value to valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value