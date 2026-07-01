def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for goal achievement indicators
    success_indicators = [
        'success', 'completed', 'saved', 'created', 'sent', 'added',
        'event created', 'task completed', 'message sent', 'confirmed',
        'done', 'finished', '✓', '✔'
    ]
    
    goal_achieved = any(ind in next_state_lower for ind in success_indicators)
    
    if goal_achieved:
        return 1.0
    
    # Action type analysis
    is_click = 'click' in action_lower
    is_fill = 'fill' in action_lower
    is_press = 'press' in action_lower
    is_noop = 'noop' in action_lower
    is_scroll = 'scroll' in action_lower
    
    # State change detection
    state_changed = state != next_state
    
    # Check for error indicators
    error_indicators = ['error', 'failed', 'invalid', 'required', 'missing', 'warning']
    has_error = any(ind in next_state_lower for ind in error_indicators)
    
    # Check for progress indicators
    progress_indicators = [
        'filled', 'entered', 'updated', 'modified', 'changed',
        'selected', 'active', 'focus', 'value'
    ]
    has_progress = any(ind in next_state_lower for ind in progress_indicators)
    
    # Check for interactive elements (potential for further progress)
    interactive_elements = ['button', 'input', 'link', 'form', 'field', 'submit']
    has_interactive = any(ind in next_state_lower for ind in interactive_elements)
    
    # Check for bid references (indicates we're interacting with elements)
    has_bid_reference = bool(re.search(r"bid['\"]?\s*[:=]\s*['\"]?\d+", action_lower))
    
    # Base Q-value calculation
    q_value = 0.0
    
    # Productive actions get higher base value
    if is_fill:
        q_value += 0.35
    elif is_click:
        q_value += 0.28
    elif is_press:
        q_value += 0.22
    elif is_scroll:
        q_value += 0.12
    
    # State change indicates progress
    if state_changed:
        q_value += 0.22
    
    # Progress indicators boost value
    if has_progress:
        q_value += 0.18
    
    # Having interactive elements means we can continue progressing
    if has_interactive:
        q_value += 0.12
    
    # Bid references suggest targeted interaction
    if has_bid_reference:
        q_value += 0.08
    
    # Error states reduce value significantly
    if has_error:
        q_value -= 0.45
    
    # Noop actions in non-goal states are less valuable
    if is_noop and not goal_achieved:
        q_value -= 0.15
    
    # Ensure Q-value is in reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value