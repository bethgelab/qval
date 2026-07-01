def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value estimate
    q_value = 0.0
    
    # Normalize to lowercase for matching
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for task completion indicators in next_state
    completion_indicators = [
        'success', 'completed', 'saved', 'added', 'sent', 'created',
        'event', 'task', 'message', 'calendar', 'todo', 'added to',
        'successfully', 'done', 'finished', 'confirmed', 'submitted',
        'posted', 'published', 'updated', 'stored'
    ]
    completion_score = sum(1 for indicator in completion_indicators 
                          if indicator in next_state_lower)
    
    # If strong completion signals, high Q-value
    if completion_score >= 2:
        return 1.0
    elif completion_score == 1:
        q_value += 0.7
    
    # Analyze action type for quality
    if 'fill' in action_lower:
        # Form filling is typically productive
        q_value += 0.15
    elif 'click' in action_lower:
        # Clicking is generally productive
        q_value += 0.1
    elif 'press' in action_lower:
        # Key presses can be productive (enter, tab, etc.)
        q_value += 0.08
    elif 'noop' in action_lower:
        # No-op actions don't progress
        q_value -= 0.05
    elif 'scroll' in action_lower:
        # Scrolling can be productive for navigation
        q_value += 0.05
    
    # Count interactive elements in states (bids indicate clickable items)
    state_bids = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?\d+", state_lower))
    next_bids = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?\d+", next_state_lower))
    
    # More visible interactive elements may indicate progress
    if next_bids > state_bids:
        q_value += 0.03 * min(5, next_bids - state_bids)
    elif next_bids < state_bids:
        q_value -= 0.01 * abs(next_bids - state_bids)
    
    # Check for error/invalid indicators in next_state
    error_keywords = ['error', 'fail', 'invalid', 'required', 'missing', 
                      'not found', 'incorrect', 'unauthorized', 'forbidden']
    error_count = sum(1 for keyword in error_keywords 
                     if keyword in next_state_lower)
    q_value -= 0.15 * error_count
    
    # Check for progress-related keywords
    progress_keywords = ['step', 'page', 'next', 'continue', 'proceed', 
                         'submit', 'save', 'add', 'create', 'send']
    progress_count = sum(1 for keyword in progress_keywords 
                        if keyword in next_state_lower)
    q_value += 0.05 * progress_count
    
    # Check if state changed meaningfully
    if len(next_state) > 0 and len(state) > 0:
        # Simple similarity check - more change might indicate progress
        state_len = len(state)
        next_len = len(next_state)
        if abs(next_len - state_len) > 100:
            q_value += 0.05
    
    # Penalize if we seem to be regressing (less content in next state)
    if next_len < state_len * 0.8:
        q_value -= 0.1
    
    # Check for specific app context indicators
    app_indicators = {
        'todo': ['todo', 'task', 'checkbox', 'checklist'],
        'calendar': ['calendar', 'event', 'date', 'time', 'schedule', 'meeting'],
        'messenger': ['message', 'chat', 'send', 'inbox', 'contact'],
        'maps': ['map', 'location', 'address', 'direction', 'search'],
        'code': ['code', 'editor', 'file', 'run', 'execute', 'output']
    }
    
    app_progress = 0
    for app, keywords in app_indicators.items():
        if any(kw in next_state_lower for kw in keywords):
            app_progress += 1
    q_value += 0.03 * app_progress
    
    # Normalize Q-value to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)