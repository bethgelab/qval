def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps web application interaction.
    
    The state-value represents expected discounted cumulative reward.
    Higher values indicate states more likely to lead to successful completion.
    """
    
    # Initialize value
    value = 0.0
    
    # Convert state to lowercase for case-insensitive matching
    state_lower = state.lower()
    
    # Check if goal is already achieved (immediate reward)
    success_keywords = ['success', 'completed', 'saved', 'added', 'sent', 'created', 'done', 'submitted']
    for keyword in success_keywords:
        if keyword in state_lower:
            return 1.0
    
    # Check for error states (negative signal)
    error_keywords = ['error', 'failed', 'invalid', 'unavailable', 'not found', 'cannot', 'unable']
    error_count = sum(1 for kw in error_keywords if kw in state_lower)
    value -= error_count * 0.15
    
    # Check for progress indicators (positive signal)
    progress_keywords = ['filled', 'entered', 'typed', 'input', 'selected', 'chosen', 'clicked']
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    value += progress_count * 0.08
    
    # Detect application type and goal-related keywords
    app_indicators = {
        'todo': ['task', 'todo', 'check', 'list', 'item', 'add task'],
        'calendar': ['event', 'date', 'time', 'schedule', 'meeting', 'reminder', 'calendar'],
        'messenger': ['message', 'chat', 'sent', 'delivered', 'contact', 'conversation', 'compose'],
        'maps': ['location', 'address', 'map', 'route', 'direction', 'search', 'place'],
        'code': ['code', 'editor', 'file', 'save', 'run', 'output', 'terminal']
    }
    
    app_found = False
    for app, keywords in app_indicators.items():
        if any(kw in state_lower for kw in keywords):
            app_found = True
            # Bonus for recognizing we're in the right app context
            value += 0.1
            break
    
    # Count interactive elements (more elements = more opportunities to act)
    bid_count = len(re.findall(r'bid["\']?\s*[:=]\s*["\']?\d+["\']?', state))
    button_count = len(re.findall(r'button', state_lower))
    input_count = len(re.findall(r'input|textbox|text', state_lower))
    
    # More interactive elements suggests more work remaining (lower value)
    # But also suggests we're in an actionable state (higher value)
    total_elements = bid_count + button_count + input_count
    if total_elements > 10:
        value += 0.1  # Rich interactive state is good
    elif total_elements < 3:
        value -= 0.1  # Sparse state might mean we're stuck
    
    # Check for navigation completeness (being on right page)
    page_keywords = ['home', 'dashboard', 'main', 'page', 'screen', 'view', 'loaded']
    page_count = sum(1 for kw in page_keywords if kw in state_lower)
    if page_count >= 2:
        value += 0.05  # Confirms we're on a proper page
    
    # Analyze state complexity as proxy for remaining work
    line_count = len(state.split('\n'))
    if line_count > 50:
        value += 0.05  # Detailed state suggests progress
    elif line_count < 10:
        value -= 0.05  # Very simple state might mean early or stuck
    
    # Penalize for potential loops (repeated elements)
    if state_lower.count('previous') > 2 or state_lower.count('back') > 2:
        value -= 0.1  # May indicate navigation issues
    
    # Check for completion hints specific to common task patterns
    completion_hints = ['confirmation', 'notification', 'alert', 'toast', 'popup']
    completion_count = sum(1 for kw in completion_hints if kw in state_lower)
    if completion_count >= 1:
        value += 0.2  # Completion feedback is a strong positive signal
    
    # Normalize value to [0, 1] range
    value = max(0.0, min(1.0, value))
    
    # Small base value for being in a valid state
    value = max(0.1, value)
    
    return round(value, 3)