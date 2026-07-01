def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Normalize strings for comparison
    state_lower = state.lower()
    next_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check if goal appears to be achieved in next_state
    success_indicators = [
        'success', 'completed', 'done', 'added', 'created',
        'saved', 'sent', 'confirmed', 'finished', '✓',
        'check', 'ok', 'yes', 'event created', 'task added',
        'message sent', 'item saved', 'appointment created'
    ]
    is_complete = any(ind in next_lower for ind in success_indicators)
    
    if is_complete:
        return 0.95
    
    # Check for error or failure indicators
    error_indicators = [
        'error', 'failed', 'invalid', 'required', 'missing',
        'wrong', 'incorrect', '×', 'close', 'cancel', 'unable'
    ]
    has_error = any(ind in next_lower for ind in error_indicators)
    
    if has_error:
        return 0.15
    
    # Analyze action type
    is_fill = 'fill' in action_lower
    is_click = 'click' in action_lower
    is_press = 'press' in action_lower
    is_noop = 'noop' in action_lower
    is_scroll = 'scroll' in action_lower
    
    # Check if state changed meaningfully (content difference)
    state_set = set(state_lower)
    next_set = set(next_lower)
    new_content = len(next_set - state_set)
    removed_content = len(state_set - next_set)
    meaningful_change = new_content > 10 or removed_content > 5
    
    # Look for progress indicators in next_state
    progress_indicators = [
        'new', 'updated', 'modified', 'changed', 'item',
        'event', 'task', 'message', 'entry', 'form',
        'field', 'input', 'selected', 'checked'
    ]
    has_progress = any(ind in next_lower for ind in progress_indicators)
    
    # Look for goal-related keywords in next_state
    goal_keywords = [
        'todo', 'calendar', 'event', 'meeting', 'appointment',
        'message', 'chat', 'map', 'location', 'code', 'editor'
    ]
    goal_context = any(kw in next_lower for kw in goal_keywords)
    
    # Base Q-value estimate
    if is_complete:
        return 0.95
    elif has_error:
        return 0.15
    elif is_noop or is_scroll:
        return 0.25
    elif not meaningful_change:
        return 0.20
    elif is_fill and meaningful_change:
        return 0.75
    elif is_click and meaningful_change:
        if has_progress:
            return 0.70
        elif goal_context:
            return 0.65
        else:
            return 0.55
    elif is_press and meaningful_change:
        return 0.60
    else:
        return 0.45