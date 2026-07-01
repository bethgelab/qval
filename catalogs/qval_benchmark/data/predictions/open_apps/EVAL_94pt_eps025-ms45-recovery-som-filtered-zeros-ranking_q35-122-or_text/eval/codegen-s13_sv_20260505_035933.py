def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment.
    
    Returns a float between 0.0 and 1.0 representing the expected
    discounted cumulative reward from this state.
    """
    import re
    
    # Convert to lowercase for pattern matching
    state_lower = state.lower()
    
    # Check for completion indicators in the state
    completion_keywords = [
        'success', 'completed', 'done', 'saved', 'created', 'added',
        'sent', 'published', 'confirmed', 'verified', 'task added',
        'event created', 'message sent', 'file saved', 'goal reached',
        'task completed', 'event saved', 'message delivered'
    ]
    
    completion_matches = sum(1 for kw in completion_keywords if kw in state_lower)
    
    # If strong completion signals, return high value
    if completion_matches >= 2:
        return 0.95
    
    # Check for specific app contexts and their success indicators
    app_indicators = {
        'todo': ['task', 'todo', 'checked', 'completed task'],
        'calendar': ['event', 'calendar', 'meeting', 'scheduled', 'date'],
        'messenger': ['message', 'sent', 'chat', 'conversation', 'delivered'],
        'maps': ['location', 'map', 'address', 'route', 'destination'],
        'code': ['file', 'saved', 'code', 'editor', 'project', 'saved file']
    }
    
    # Identify which app context we're in and count relevant indicators
    app_score = 0
    for app, indicators in app_indicators.items():
        indicator_count = sum(1 for ind in indicators if ind in state_lower)
        if indicator_count > app_score:
            app_score = indicator_count
    
    # Estimate progress based on available actions and elements
    # Count interactive elements that could advance the task
    button_count = len(re.findall(r'button|click|submit|save|add|create', state_lower))
    form_count = len(re.findall(r'input|form|text|field|enter', state_lower))
    
    # Check for task-specific content that indicates progress
    progress_indicators = [
        'step', 'progress', 'loading', 'working', 'processing',
        'adding', 'creating', 'sending', 'opening', 'editing'
    ]
    progress_count = sum(1 for ind in progress_indicators if ind in state_lower)
    
    # Calculate value based on progress indicators
    base_value = 0.1  # Starting value for any non-terminal state
    
    # Add value for completion indicators (strongest signal)
    base_value += completion_matches * 0.15
    
    # Add value for app-specific progress
    base_value += app_score * 0.08
    
    # Add value for available actions (more actions = more flexibility)
    base_value += min(button_count * 0.03, 0.25)
    base_value += min(form_count * 0.02, 0.15)
    
    # Add value for active progress indicators
    base_value += progress_count * 0.05
    
    # Check for error or failure indicators (reduce value)
    error_keywords = ['error', 'failed', 'invalid', 'unavailable', 'not found', 'blocked']
    error_count = sum(1 for kw in error_keywords if kw in state_lower)
    base_value -= error_count * 0.1
    
    # Check for empty or initial state indicators (lower value)
    empty_keywords = ['empty', 'no tasks', 'no events', 'no messages', 'blank', 'initial']
    empty_count = sum(1 for kw in empty_keywords if kw in state_lower)
    base_value -= empty_count * 0.05
    
    # Clamp to valid range
    return min(max(base_value, 0.0), 1.0)