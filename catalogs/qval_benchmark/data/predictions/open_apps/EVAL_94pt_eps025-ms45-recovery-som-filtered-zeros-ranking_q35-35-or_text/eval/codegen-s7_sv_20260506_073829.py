import re
from collections import Counter

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for success indicators - goal already achieved
    success_patterns = [
        'goal achieved', 'task complete', 'success', 'completed',
        'event added', 'message sent', 'task created', 'saved',
        'verified', 'confirmed', 'done'
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            return 1.0
    
    # Check for error states that indicate poor outcomes
    error_patterns = ['error', 'failed', 'invalid', 'not found', 'cannot', 'blocked']
    error_count = sum(1 for pattern in error_patterns if pattern in state_lower)
    if error_count >= 2:
        return 0.15
    
    # Identify app context and relevant goal indicators
    goal_keywords = {
        'todo': ['task', 'todo', 'list', 'add', 'create', 'complete'],
        'calendar': ['event', 'calendar', 'date', 'schedule', 'meeting'],
        'messenger': ['message', 'chat', 'send', 'reply', 'compose'],
        'maps': ['directions', 'location', 'navigate', 'route', 'map'],
        'code': ['code', 'editor', 'save', 'run', 'file', 'script']
    }
    
    # Count relevant goal-related elements present
    goal_match_count = 0
    for app, keywords in goal_keywords.items():
        for keyword in keywords:
            if keyword in state_lower:
                goal_match_count += 1
    
    # Count interactive elements available (bids indicate clickable elements)
    bid_matches = re.findall(r'bid\s*[=:]\s*\d+', state_lower)
    interactive_count = len(bid_matches)
    
    # Check for action buttons that advance the task
    action_buttons = ['submit', 'save', 'send', 'add', 'create', 'confirm', 
                      'ok', 'yes', 'done', 'next', 'back', 'cancel']
    action_count = sum(1 for btn in action_buttons if btn in state_lower)
    
    # Check for form fields that need filling (progress indicator)
    form_indicators = ['input', 'text', 'field', 'label', 'placeholder', 
                       'name', 'email', 'password', 'date', 'time']
    form_count = sum(1 for ind in form_indicators if ind in state_lower)
    
    # Calculate base value from multiple features
    base_value = 0.5
    
    # More interactive elements = better (can take actions)
    base_value += min(0.25, interactive_count * 0.05)
    
    # Goal-related elements present = on right track
    base_value += min(0.2, goal_match_count * 0.02)
    
    # Available action buttons = can make progress
    base_value += min(0.15, action_count * 0.03)
    
    # Form fields present = need to complete task (moderate value)
    if form_count > 0:
        base_value += 0.05
    
    # Penalty for errors
    base_value -= error_count * 0.1
    
    # Bonus if we're in a task-appropriate context
    if goal_match_count >= 2:
        base_value += 0.1
    
    # Clamp to valid range
    return max(0.0, min(1.0, base_value))