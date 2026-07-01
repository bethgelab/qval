import re
import math

def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment based on state representation.
    Returns expected discounted cumulative reward (0.0 to 1.0).
    """
    # Initialize features
    goal_achieved = False
    app_context = ""
    forms_filled = 0
    total_forms = 0
    has_error = False
    has_submit_button = False
    has_navigation = False
    goal_indicators = 0
    
    # Check if goal already achieved
    if 'goal' in state.lower() and ('achieved' in state.lower() or 'complete' in state.lower()):
        goal_achieved = True
        return 1.0
    
    # Identify app context
    app_keywords = {
        'todo': ['todo', 'task', 'list', 'item'],
        'calendar': ['calendar', 'event', 'date', 'schedule'],
        'messenger': ['message', 'chat', 'send', 'inbox'],
        'maps': ['map', 'location', 'address', 'route'],
        'code': ['code', 'editor', 'file', 'script']
    }
    
    for app, keywords in app_keywords.items():
        for keyword in keywords:
            if keyword in state.lower():
                app_context = app
                break
    
    # Count form fields and filled status
    field_patterns = [
        r'input[^>]*',
        r'textarea[^>]*',
        r'fillable[^>]*',
        r'field[^>]*',
        r'editable[^>]*'
    ]
    
    for pattern in field_patterns:
        matches = re.findall(pattern, state, re.IGNORECASE)
        total_forms += len(matches)
    
    # Check for filled forms
    filled_patterns = [
        r'value\s*=\s*["\'][^"\']+["\']',
        r'filled',
        r'selected',
        r'text\s*:\s*["\'][^"\']+["\']'
    ]
    
    for pattern in filled_patterns:
        matches = re.findall(pattern, state, re.IGNORECASE)
        if matches:
            forms_filled += len(matches)
    
    # Check for errors
    error_patterns = [
        r'error',
        r'invalid',
        r'unable',
        r'failed',
        r'not found'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            has_error = True
            break
    
    # Check for submit/complete buttons
    submit_patterns = [
        r'submit',
        r'save',
        r'complete',
        r'finish',
        r'confirm',
        r'send',
        r'add',
        r'create'
    ]
    
    for pattern in submit_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            has_submit_button = True
            break
    
    # Check for navigation elements
    nav_patterns = [
        r'link',
        r'click',
        r'button',
        r'navigate',
        r'go to'
    ]
    
    for pattern in nav_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            has_navigation = True
            break
    
    # Check for goal-related indicators
    goal_patterns = [
        r'add event',
        r'create task',
        r'send message',
        r'find location',
        r'write code',
        r'goal',
        r'task'
    ]
    
    for pattern in goal_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            goal_indicators += 1
    
    # Calculate progress score
    progress = 0.0
    
    # Base progress from app context identification
    if app_context:
        progress += 0.1
    
    # Progress from form completion
    if total_forms > 0:
        progress += min(0.3, forms_filled / total_forms * 0.3)
    
    # Progress from having submit button (ready to complete)
    if has_submit_button:
        progress += 0.2
    
    # Progress from navigation (moving toward goal)
    if has_navigation:
        progress += 0.1
    
    # Progress from goal indicators
    progress += min(0.2, goal_indicators * 0.05)
    
    # Penalty for errors
    if has_error:
        progress = max(0.0, progress - 0.2)
    
    # Bonus for being near completion (many goal indicators, submit button, forms filled)
    completion_score = 0.0
    if has_submit_button and forms_filled > 0:
        completion_score += 0.2
    if goal_indicators >= 2:
        completion_score += 0.1
    
    progress += completion_score
    
    # Discount based on steps remaining (estimate)
    # If we're early in the task, value is lower; late stage, higher
    step_estimate = 15 + goal_indicators * 5  # rough estimate of steps needed
    
    # Exponential decay for remaining steps
    discount = math.exp(-step_estimate / 30)
    
    # Final value
    value = min(1.0, max(0.0, progress * discount + 0.1))
    
    return value