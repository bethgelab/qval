import re

def signal_function(state: str) -> float:
    if not state:
        return 0.0
    
    # Normalize state for case-insensitive matching
    text = state.lower()
    
    # 1. Check for Goal Achievement (Value = 1.0)
    success_indicators = [
        'task completed', 'event added', 'message sent', 'file saved', 
        'done', 'success', 'completed', 'created', 'added', 'sent', 
        'route found', 'found', 'displayed', 'visible', 'saved'
    ]
    for indicator in success_indicators:
        if indicator in text:
            return 1.0
    
    # 2. Check for Errors/Dead Ends (Value = 0.0)
    error_indicators = [
        'error', 'failed', 'unable', 'timeout', 'cannot', 'not found', '404',
        'disabled', 'read only', 'pending', 'loading', 'waiting', 
        'processing', 'connecting'
    ]
    for indicator in error_indicators:
        if indicator in text:
            return 0.0
    
    # 3. Calculate Progress Score
    score = 0.0
    
    # High Priority: Actionable Buttons (Submit, Send, Add, Save)
    action_indicators = ['submit', 'send', 'add', 'create', 'save', 'confirm', 'ok', 'yes', 'button', 'click', 'ready']
    action_count = sum(1 for ind in action_indicators if ind in text)
    score += min(action_count * 0.15, 0.5)
    
    # Medium Priority: Input Fields (Title, Date, Message, Recipient)
    input_indicators = ['input', 'text', 'field', 'title', 'date', 'recipient', 'message']
    input_count = sum(1 for ind in input_indicators if ind in text)
    score += min(input_count * 0.05, 0.2)
    
    # Low Priority: Navigation/Lists (Home, List, Menu, Dashboard)
    nav_indicators = ['home', 'list', 'menu', 'dashboard', 'open', 'back', 'forward']
    nav_count = sum(1 for ind in nav_indicators if ind in text)
    score += min(nav_count * 0.02, 0.1)
    
    # Bonus for specific app context presence (indicates correct app)
    app_context = ['todo', 'calendar', 'messenger', 'maps', 'code']
    app_count = sum(1 for ind in app_context if ind in text)
    score += min(app_count * 0.05, 0.1)
    
    # Ensure score is within [0, 1]
    return float(min(max(score, 0.0), 1.0))