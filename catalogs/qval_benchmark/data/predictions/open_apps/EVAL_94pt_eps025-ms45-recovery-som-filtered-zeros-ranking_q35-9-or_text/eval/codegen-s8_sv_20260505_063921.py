import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for success indicators (goal achieved or nearly achieved)
    success_patterns = ['completed', 'success', 'done', 'saved', 'created', 'added', 
                        'sent', 'scheduled', 'message sent', 'event created', 
                        'task complete', 'goal achieved', 'verification passed']
    for pattern in success_patterns:
        if pattern in state_lower:
            return 0.95
    
    # Check for error/failure indicators
    failure_patterns = ['error', 'failed', 'invalid', 'missing', 'unavailable', 
                        'not found', 'permission denied', 'blocked', 'timeout',
                        'cannot', 'unable', 'failed to']
    failure_count = 0
    for pattern in failure_patterns:
        if pattern in state_lower:
            failure_count += 1
    if failure_count >= 2:
        return 0.1
    
    # Check if we're on a form page (likely closer to goal completion)
    form_indicators = ['input', 'text', 'email', 'password', 'submit', 'button', 
                       'checkbox', 'radio', 'select', 'textarea', 'form']
    form_count = len([p for p in form_indicators if p in state_lower])
    
    # Check for goal-specific keywords
    goal_keywords = ['calendar', 'event', 'message', 'todo', 'code', 'map',
                     'appointment', 'meeting', 'contact', 'task', 'note']
    goal_keywords_count = len([k for k in goal_keywords if k in state_lower])
    
    # Check for progress indicators
    progress_patterns = ['step', 'progress', 'remaining', 'current', 'next',
                         'stage', 'phase', 'completion', 'percentage']
    progress_count = len([p for p in progress_patterns if p in state_lower])
    
    # Check for navigation status
    nav_patterns = ['home', 'page', 'tab', 'menu', 'link', 'navigation', 'back',
                    'forward', 'next', 'previous', 'scroll', 'section']
    nav_count = len([n for n in nav_patterns if n in state_lower])
    
    # Calculate base value based on features
    base_value = 0.5
    
    # Reward for being on a form page
    if form_count >= 3:
        base_value += 0.15
    elif form_count >= 1:
        base_value += 0.05
    
    # Reward for goal-related content
    if goal_keywords_count >= 2:
        base_value += 0.15
    elif goal_keywords_count == 1:
        base_value += 0.05
    
    # Reward for progress indicators
    if progress_count >= 2:
        base_value += 0.1
    elif progress_count == 1:
        base_value += 0.05
    
    # Reward for navigation clarity (many options = more flexibility)
    if nav_count >= 3:
        base_value += 0.05
    
    # Penalty for high failure indicators
    if failure_count >= 1:
        base_value -= 0.1
    
    # Cap value between 0 and 1
    return max(0.0, min(1.0, base_value))