import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    state_words = re.findall(r'\w+', state.lower())
    next_words = re.findall(r'\w+', next_state.lower())
    
    state_count = len(set(state_words))
    next_count = len(set(next_words))
    
    action_type = action.split('(')[0].strip() if '(' in action else 'unknown'
    
    q_value = 0.0
    
    if action_type in ['click', 'fill', 'press', 'scroll']:
        q_value += 0.15
    elif action_type == 'noop':
        q_value -= 0.1
    
    if next_count > state_count:
        q_value += 0.1
    elif next_count < state_count:
        q_value -= 0.05
    
    completion_keywords = ['success', 'complete', 'done', 'saved', 'added', 'created', 
                          'event', 'message', 'todo', 'calendar', 'map', 'code']
    if any(keyword in next_state.lower() for keyword in completion_keywords):
        q_value += 0.25
    
    error_keywords = ['error', 'failed', 'invalid', 'missing', 'unavailable']
    if any(keyword in next_state.lower() for keyword in error_keywords):
        q_value -= 0.3
    
    if any(keyword in next_state.lower() for keyword in completion_keywords):
        q_value += 0.15
    
    if 'bid' in action.lower() and '(' in action:
        bid_match = re.search(r'\([^)]+\)', action)
        if bid_match and len(bid_match.group()) > 5:
            q_value += 0.05
    
    if 'fill' in action_type.lower():
        if 'text' in action.lower():
            q_value += 0.08
    
    q_value = max(0.0, min(1.0, q_value))
    return q_value