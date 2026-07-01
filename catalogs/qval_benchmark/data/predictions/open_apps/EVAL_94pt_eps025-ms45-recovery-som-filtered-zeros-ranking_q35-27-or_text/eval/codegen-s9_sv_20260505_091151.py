import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Start with a baseline value (moderate optimism)
    value = 0.25
    
    # 1. Check for goal achievement indicators (strongest positive signal)
    success_keywords = [
        'success', 'completed', 'created', 'added', 'saved', 
        'sent', 'submitted', 'confirmed', 'done', 'finished',
        'event created', 'task added', 'message sent', 'calendar event'
    ]
    success_count = sum(1 for kw in success_keywords if kw in state_lower)
    if success_count > 0:
        value = min(1.0, 0.85 + success_count * 0.08)
    
    # 2. Check for error/failure indicators (strong negative signal)
    error_keywords = [
        'error', 'failed', 'invalid', 'cannot', 'not found',
        'does not exist', 'unable', 'problem', 'issue', 'warning'
    ]
    error_count = sum(1 for kw in error_keywords if kw in state_lower)
    if error_count > 0:
        value = max(0.0, value - error_count * 0.18)
    
    # 3. Check for progress indicators (moderate positive signal)
    progress_keywords = [
        'filled', 'entered', 'input', 'text', 'field', 'value',
        'submit', 'button', 'next', 'continue', 'save', 'create',
        'form', 'data', 'content'
    ]
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    if progress_count > 0:
        value = min(1.0, value + progress_count * 0.04)
    
    # 4. Check for relevant app context (we're in the right application)
    context_keywords = [
        'calendar', 'event', 'todo', 'task', 'message', 'messenger',
        'contact', 'map', 'location', 'code', 'editor', 'file', 'app'
    ]
    context_count = sum(1 for kw in context_keywords if kw in state_lower)
    if context_count > 0:
        value = min(1.0, value + context_count * 0.025)
    
    # 5. Count available interactive elements (bids) - indicates available actions
    bid_matches = re.findall(r'bid\s*[=:]\s*\d+', state_lower)
    bid_count = len(bid_matches)
    if bid_count > 0:
        # Moderate number of bids indicates good action availability
        if 5 <= bid_count <= 25:
            value = min(1.0, value + 0.1)
        elif bid_count < 5:
            value = max(0.0, value - 0.05)  # Too few options may indicate stuck state
    
    # 6. Check for step information (efficiency matters with 45 step limit)
    step_matches = re.findall(r'step\s*[=:]\s*(\d+)', state_lower)
    if step_matches:
        steps = int(step_matches[0])
        # Penalize if too many steps have been taken
        if steps > 35:
            value = max(0.0, value - 0.25)
        elif steps > 25:
            value = max(0.0, value - 0.1)
        elif steps < 10:
            value = min(1.0, value + 0.05)  # Early progress is favorable
    
    # 7. Check for filled form fields (indicates task progress)
    filled_pattern = r'value\s*[=:]\s*["\']?[^\s"\']+'
    filled_count = len(re.findall(filled_pattern, state_lower))
    if filled_count > 0:
        value = min(1.0, value + filled_count * 0.02)
    
    # Clamp to valid range [0, 1]
    value = max(0.0, min(1.0, value))
    
    return value