import re
from collections import Counter

def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment based on state features.
    Returns a value between 0.0 and 1.0+ indicating how favorable the state is.
    """
    state_lower = state.lower()
    
    # Base value
    value = 0.5
    
    # Check if goal has been achieved (reward 1.0)
    achievement_patterns = [
        r'task\s*completed',
        r'goal\s*achieved',
        r'success',
        r'confirmed',
        r'added',
        r'sent',
        r'saved',
        r'created',
        r'updated',
        r'done',
    ]
    for pattern in achievement_patterns:
        if re.search(pattern, state_lower):
            return 1.0
    
    # Check for error states (reduce value)
    error_patterns = [
        r'error',
        r'failed',
        r'unable',
        r'denied',
        r'not\s*found',
        r'missing',
        r'invalid',
        r'cannot',
    ]
    error_count = sum(1 for p in error_patterns if re.search(p, state_lower))
    value -= error_count * 0.15
    
    # Check if we're on the correct app/page for tasks
    app_patterns = {
        'calendar': [r'calendar', r'event', r'date', r'appointment', r'agenda'],
        'todo': [r'todo', r'task', r'task\s*list', r'checklist', r'item'],
        'messenger': [r'message', r'chat', r'messenger', r'send', r'conversation'],
        'maps': [r'map', r'location', r'address', r'route', r'place'],
        'code': [r'code', r'editor', r'file', r'edit', r'syntax'],
    }
    
    app_matches = Counter()
    for app, patterns in app_patterns.items():
        for pattern in patterns:
            if re.search(pattern, state_lower):
                app_matches[app] += 1
    
    # Bonus for being on relevant app pages
    if app_matches:
        value += min(len(app_matches) * 0.05, 0.2)
    
    # Check for interactive elements that enable progress
    interactive_patterns = [
        r'button',
        r'click',
        r'clickable',
        r'input',
        r'form',
        r'text\s*field',
        r'submit',
        r'create',
        r'add',
        r'new',
    ]
    interactive_count = sum(1 for p in interactive_patterns if re.search(p, state_lower))
    value += min(interactive_count * 0.03, 0.15)
    
    # Check for navigation elements (good for reaching goal)
    nav_patterns = [r'home', r'back', r'forward', r'menu', r'nav', r'link', r'navigate']
    nav_count = sum(1 for p in nav_patterns if re.search(p, state_lower))
    value += min(nav_count * 0.02, 0.1)
    
    # Check for loading states (bad - waiting)
    if re.search(r'loading|loading...', state_lower):
        value -= 0.1
    if re.search(r'processing|processing...', state_lower):
        value -= 0.05
    
    # Check for empty/blank states (bad - nothing to work with)
    if re.search(r'empty|nothing|no\s*items|no\s*results', state_lower):
        value -= 0.1
    
    # Check for step progression indicators
    step_patterns = [r'step\s*\d+', r'progress', r'1\s*of', r'step\s*1', r'step\s*2']
    step_matches = sum(1 for p in step_patterns if re.search(p, state_lower))
    value += min(step_matches * 0.03, 0.09)
    
    # Check if we're at the start (neutral to slightly negative)
    if re.search(r'welcome|start|begin|landing|home', state_lower):
        value -= 0.05
    
    # Clamp value to reasonable range
    value = max(0.0, min(1.5, value))
    
    return value