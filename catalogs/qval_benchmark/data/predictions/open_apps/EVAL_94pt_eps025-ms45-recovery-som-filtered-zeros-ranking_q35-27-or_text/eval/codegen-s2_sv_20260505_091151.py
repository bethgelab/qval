import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Strong positive indicators - task likely complete
    completion_phrases = [
        'created successfully', 'added successfully', 'saved successfully',
        'sent successfully', 'completed', 'success', 'done', 'finished',
        'confirmed', 'event created', 'message sent', 'todo added',
        'task completed', 'calendar event', 'new event'
    ]
    
    # Moderate positive indicators - relevant UI elements present
    progress_indicators = [
        'calendar', 'event', 'todo', 'task', 'message', 'messenger',
        'contact', 'maps', 'location', 'address', 'editor', 'code',
        'form', 'input', 'button', 'submit', 'save', 'create',
        'bid', 'click', 'fill', 'press', 'noop', 'scroll'
    ]
    
    # Negative indicators - errors or problems
    error_indicators = [
        'error', 'failed', 'invalid', 'missing', 'required',
        'cannot', 'unable', 'denied', 'not found', 'unavailable',
        'timeout', 'connection', 'refused', 'blocked'
    ]
    
    score = 0.0
    
    # Check completion indicators (strong positive)
    for phrase in completion_phrases:
        if phrase in state_lower:
            score += 0.35
            break
    
    # Check progress indicators (moderate positive)
    progress_count = sum(1 for ind in progress_indicators if ind in state_lower)
    score += min(progress_count * 0.04, 0.4)
    
    # Check error indicators (negative)
    error_count = sum(1 for ind in error_indicators if ind in state_lower)
    score -= min(error_count * 0.15, 0.5)
    
    # Check for bid numbers (indicates interactive elements available)
    if re.search(r'bid[\'"]?\s*=\s*[\'"]?\d+', state_lower):
        score += 0.1
    
    # Check for form fields being filled (positive signal)
    if re.search(r'value\s*=', state_lower) or re.search(r'filled', state_lower):
        score += 0.08
    
    # Ensure value is in [0, 1] range
    score = max(0.0, min(1.0, score))
    
    return float(score)