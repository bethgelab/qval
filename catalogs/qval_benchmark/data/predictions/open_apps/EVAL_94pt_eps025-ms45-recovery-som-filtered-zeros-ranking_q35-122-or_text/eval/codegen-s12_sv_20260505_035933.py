def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for explicit success/completion indicators
    success_patterns = ['success', 'completed', 'saved', 'added', 'sent', 'created', 
                       'task added', 'event created', 'message sent', 'note saved',
                       'successfully', 'done', 'confirmed']
    success_score = sum(1 for pattern in success_patterns if pattern in state_lower)
    
    # Check for task progress indicators
    progress_patterns = ['task', 'event', 'message', 'note', 'item', 'entry', 'list',
                        'todo', 'calendar', 'messenger', 'map', 'code']
    progress_score = sum(1 for pattern in progress_patterns if pattern in state_lower)
    
    # Count interactive elements (bid tags indicate clickable/fillable elements)
    bid_matches = re.findall(r"bid['\"]?\s*[:=]?\s*['\"]?(\d+)", state)
    bid_count = len(bid_matches)
    
    # Check for error/negative states
    error_patterns = ['error', 'failed', 'invalid', 'missing', 'required', 
                     'please fill', 'cannot', 'unable', 'unavailable', 'not found']
    error_score = sum(1 for pattern in error_patterns if pattern in state_lower)
    
    # Check for form completion indicators
    form_patterns = ['filled', 'entered', 'typed', 'input', 'value', 'selected']
    form_score = sum(1 for pattern in form_patterns if pattern in state_lower)
    
    # Check for navigation elements (indicates ability to move through app)
    nav_patterns = ['link', 'button', 'click', 'navigate', 'page', 'menu', 'tab']
    nav_score = sum(1 for pattern in nav_patterns if pattern in state_lower)
    
    # Calculate value estimate
    value = 0.0
    
    # Success indicators contribute most to value (strongest signal)
    value += min(0.7, success_score * 0.2)
    
    # Progress indicators suggest we're on the right track
    value += min(0.2, progress_score * 0.02)
    
    # More interactive elements = more navigation options and flexibility
    value += min(0.1, bid_count * 0.002)
    
    # Form completion suggests progress toward goal
    value += min(0.08, form_score * 0.02)
    
    # Navigation elements indicate ability to explore/approach goal
    value += min(0.05, nav_score * 0.01)
    
    # Error indicators reduce value significantly
    value -= min(0.4, error_score * 0.15)
    
    # Check for specific app contexts to add context-aware value
    if 'todo' in state_lower or 'task' in state_lower:
        value += 0.02
    if 'calendar' in state_lower or 'event' in state_lower:
        value += 0.02
    if 'messenger' in state_lower or 'message' in state_lower:
        value += 0.02
    if 'maps' in state_lower or 'map' in state_lower:
        value += 0.02
    if 'code' in state_lower or 'editor' in state_lower:
        value += 0.02
    
    # Bonus for having multiple progress signals (indicates active task execution)
    if success_score > 0 and form_score > 0:
        value += 0.05
    
    # Ensure value stays in valid range [0, 1]
    value = max(0.0, min(1.0, value))
    
    return value