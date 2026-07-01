import re
import math

def signal_function(state: str) -> float:
    value = 0.5
    
    goal_indicators = ['goal achieved', 'success', 'completed', 'done', 'task finished']
    error_indicators = ['error', 'failed', 'invalid', 'not found', 'unable', 'missing']
    form_indicators = ['form', 'input', 'text', 'email', 'password', 'date', 'time']
    nav_indicators = ['home', 'page', 'link', 'button', 'navigate', 'back', 'forward']
    bid_pattern = r'bid\d+'
    
    if any(ind in state.lower() for ind in goal_indicators):
        return 1.0
    
    if any(ind in state.lower() for ind in error_indicators):
        value = 0.1
    
    bid_matches = re.findall(bid_pattern, state)
    bid_count = len(bid_matches)
    
    form_matches = re.findall(r'\b(form|input|text|email|password|date|time|checkbox|radio)\b', state.lower())
    form_count = len(form_matches)
    
    if form_count >= 2:
        value = min(value + 0.2, 0.9)
    elif form_count == 1:
        value = min(value + 0.1, 0.8)
    
    if bid_count >= 10:
        value = min(value + 0.15, 0.95)
    elif bid_count >= 5:
        value = min(value + 0.1, 0.9)
    elif bid_count >= 2:
        value = min(value + 0.05, 0.85)
    
    if 'loading' in state.lower() or 'loading' in state.lower():
        value = 0.3
    
    if 'home' in state.lower() and 'page' in state.lower():
        value = 0.4
    
    if 'calendar' in state.lower() or 'todo' in state.lower() or 'messenger' in state.lower():
        value = min(value + 0.1, 0.9)
    
    if 'code' in state.lower() and 'editor' in state.lower():
        value = min(value + 0.05, 0.85)
    
    if 'map' in state.lower():
        value = min(value + 0.05, 0.85)
    
    return max(0.0, min(1.0, value))