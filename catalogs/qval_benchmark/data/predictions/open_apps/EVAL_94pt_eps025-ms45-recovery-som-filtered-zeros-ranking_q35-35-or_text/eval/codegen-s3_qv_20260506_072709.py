def signal_function(state: str, action: str, next_state: str) -> float:
    """Estimate Q-value for OpenApps browser automation task."""
    
    # Check for goal completion indicators
    goal_patterns = [
        'completed', 'added', 'created', 'sent', 'saved', 'done',
        'success', 'confirmed', 'event created', 'message sent',
        'task completed', 'calendar event', 'todo added', 'task done'
    ]
    next_lower = next_state.lower()
    
    if any(pattern in next_lower for pattern in goal_patterns):
        return 1.0
    
    # Evaluate action types and their typical outcomes
    action_lower = action.lower()
    
    # High-value actions that typically advance toward goals
    if any(
        pat in action_lower 
        for pat in ['click.*button', 'fill.*submit', 'press.*enter']
    ):
        return 0.4
    
    # Moderate progress actions
    if any(
        pat in action_lower 
        for pat in ['click', 'fill', 'press']
    ):
        return 0.3
    
    # Low-value actions (no-op, scroll without progress)
    if any(
        pat in action_lower 
        for pat in ['noop', 'scroll']
    ):
        return 0.1
    
    # Check for error or failure states
    error_patterns = [
        'error', 'invalid', 'not found', 'failed', 'unable',
        'cannot', 'missing', 'empty', 'permission denied'
    ]
    if any(pattern in next_lower for pattern in error_patterns):
        return -0.3
    
    # Default moderate value for unclassified states
    return 0.2