import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check if goal is already achieved (highest priority)
    success_patterns = ['goal achieved', 'task complete', 'success', 'completed', 'done', 
                        'mission accomplished', 'all tasks completed', 'message sent', 
                        'event added', 'calendar event created', 'task added']
    if any(pattern in state_lower for pattern in success_patterns):
        return 1.0
    
    # Check for error/failure states (low value)
    error_patterns = ['error', 'failed', 'not found', '404', 'invalid input', 
                      'unable to', 'connection failed', 'timeout']
    if any(pattern in state_lower for pattern in error_patterns):
        return 0.1
    
    # Check for navigation to wrong app/context
    wrong_context_patterns = ['login', 'sign in', 'authentication', 'permission denied']
    if any(pattern in state_lower for pattern in wrong_context_patterns):
        return 0.15
    
    # Check current app context relevance
    app_contexts = {
        'calendar': ['calendar', 'event', 'date', 'time', 'appointment', 'schedule'],
        'todo': ['todo', 'task', 'list', 'checklist', 'add task', 'create task'],
        'messenger': ['message', 'chat', 'send', 'conversation', 'contact', 'inbox'],
        'maps': ['map', 'location', 'address', 'directions', 'route', 'navigate'],
        'code': ['code', 'editor', 'file', 'script', 'function', 'compile']
    }
    
    app_score = 0.0
    for app, indicators in app_contexts.items():
        if any(ind in state_lower for ind in indicators):
            app_score = 0.35
            break
    
    # Check for progress indicators
    progress_patterns = ['step', '1 of', '2 of', 'page', 'section', 'form', 'field']
    if any(p in state_lower for p in progress_patterns):
        app_score += 0.15
    
    # Check for interactive elements that enable progress
    interactive_elements = ['button', 'input', 'form', 'link', 'submit', 'save', 
                           'add', 'create', 'send', 'click', 'fill', 'textarea']
    if any(elem in state_lower for elem in interactive_elements):
        app_score += 0.25
    
    # Check for action availability
    action_patterns = ['add', 'create', 'send', 'save', 'submit', 'new', 'start', 'begin']
    if any(p in state_lower for p in action_patterns):
        app_score += 0.2
    
    # Check for navigation breadcrumbs (indicates path clarity)
    nav_patterns = ['home', 'back', 'next', 'previous', 'menu', 'navigation', 'breadcrumb']
    if any(p in state_lower for p in nav_patterns):
        app_score += 0.1
    
    # Cap and return score
    return min(1.0, max(0.0, app_score))