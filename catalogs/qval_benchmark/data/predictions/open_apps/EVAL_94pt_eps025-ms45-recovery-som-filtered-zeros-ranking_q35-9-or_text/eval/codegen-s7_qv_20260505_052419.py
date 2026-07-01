import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    # Check if action is productive (click, fill, press vs noop, scroll)
    if 'noop' in action.lower():
        q_value -= 0.3
    elif 'scroll' in action.lower():
        q_value -= 0.1
    elif 'click' in action.lower() or 'fill' in action.lower() or 'press' in action.lower():
        q_value += 0.3
    
    # Check for bid targeting (interactive elements)
    if re.search(r'bid\s*[\'"]?\d+[\'"]?', action):
        q_value += 0.2
    
    # Check if state contains goal-related keywords
    goal_keywords = ['calendar', 'todo', 'message', 'map', 'code', 'event', 'task', 'complete', 'done', 'success', 'goal', 'add', 'send', 'create', 'edit', 'delete']
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    goal_count_state = sum(1 for kw in goal_keywords if kw in state_lower)
    goal_count_next = sum(1 for kw in goal_keywords if kw in next_state_lower)
    
    if goal_count_state > 0:
        q_value += 0.15 * goal_count_state
    
    # Check for completion indicators in next_state
    completion_indicators = ['completed', 'done', 'success', 'goal', 'added', 'saved', 'created', 'sent', 'message sent', 'event added', 'task completed', 'goal reached']
    for indicator in completion_indicators:
        if indicator in next_state_lower:
            q_value += 0.5
            break
    
    # Check if next_state shows progress (more content, different page)
    if len(next_state) > len(state) and len(next_state) > 100:
        q_value += 0.1
    
    # Check for error/failure indicators
    error_indicators = ['error', 'failed', 'invalid', 'not found', 'cannot', 'unable', 'missing']
    if any(ind in next_state_lower for ind in error_indicators):
        q_value -= 0.4
    
    # Check for form completion indicators
    if 'bid' in action.lower() and ('fill' in action.lower() or 'click' in action.lower()):
        if 'submit' in next_state_lower or 'save' in next_state_lower:
            q_value += 0.25
    
    # Consider if next_state appears more complete than state
    if 'bid' in next_state_lower and next_state_lower.count('bid') > state_lower.count('bid'):
        q_value += 0.1
    
    # Clamp to reasonable range
    q_value = max(-0.5, min(1.0, q_value))
    
    return q_value