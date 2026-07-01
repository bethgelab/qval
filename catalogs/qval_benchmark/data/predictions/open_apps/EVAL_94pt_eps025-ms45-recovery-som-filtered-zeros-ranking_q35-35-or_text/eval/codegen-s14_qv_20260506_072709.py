import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    # Check if goal is already achieved in the current state
    goal_patterns = [
        r'goal.*completed|task.*completed|goal.*achieved|success',
        r'event.*added|calendar.*event|message.*sent|todo.*added',
        r'navigation.*complete|page.*loaded|form.*submitted',
        r'✓|success|done|completed|finished'
    ]
    
    for pattern in goal_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 1.0
    
    # Check if we're in an error/failure state
    error_patterns = [
        r'error|failed|invalid|not.*found|unauthorized|denied',
        r'404|403|500|timeout|expired'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 0.0
    
    # Analyze action appropriateness
    action_value = 0.5  # Default neutral value
    
    # Good actions for navigation
    good_nav_actions = ['click', 'press', 'scroll']
    if any(nav in action.lower() for nav in good_nav_actions):
        action_value += 0.1
    
    # Good actions for form completion
    good_form_actions = ['fill']
    if any(form in action.lower() for form in good_form_actions):
        action_value += 0.15
    
    # Bad actions (noop when progress needed)
    if 'noop' in action.lower():
        action_value -= 0.1
    
    # Analyze progress from state to next_state
    progress_value = 0.0
    
    # Check for state changes that indicate progress
    state_words = set(re.findall(r'\b\w+\b', state.lower()))
    next_state_words = set(re.findall(r'\b\w+\b', next_state.lower()))
    
    # Progress indicators
    progress_keywords = ['event', 'message', 'todo', 'calendar', 'map', 'page', 'form', 'item', 'entry']
    
    progress_count = 0
    for keyword in progress_keywords:
        if keyword in next_state_words and keyword not in state_words:
            progress_count += 1
    
    if progress_count > 0:
        progress_value = min(0.3, progress_count * 0.1)
    
    # Check if next state has more interactive elements (indicating navigation)
    bid_count_state = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?\d+['\"]?", state))
    bid_count_next = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?\d+['\"]?", next_state))
    
    if bid_count_next > bid_count_state:
        progress_value += 0.1
    elif bid_count_next < bid_count_state:
        progress_value -= 0.1
    
    # Check if next state is closer to completion
    next_state_goal_patterns = [
        r'goal.*completed|task.*completed|goal.*achieved',
        r'event.*added|calendar.*event|message.*sent|todo.*added',
        r'✓|success|done|completed|finished'
    ]
    
    next_goal_reached = False
    for pattern in next_state_goal_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            next_goal_reached = True
            break
    
    if next_goal_reached:
        return 1.0
    
    # Combine all factors
    q_value = 0.3 + action_value + progress_value
    
    # Cap at reasonable range
    q_value = max(0.0, min(1.0, q_value))
    
    # Bonus for actions that seem to directly advance toward goal
    action_goal_alignment = 0.0
    
    # Click on submit/confirm/save buttons is good
    if any(term in action.lower() for term in ['submit', 'confirm', 'save', 'add', 'send', 'create']):
        action_goal_alignment += 0.2
    
    # Filling form fields is good
    if 'fill' in action.lower():
        action_goal_alignment += 0.15
    
    # Clicking on navigation elements is good
    if any(term in action.lower() for term in ['navigate', 'go', 'back', 'forward', 'home']):
        action_goal_alignment += 0.1
    
    q_value = min(1.0, q_value + action_goal_alignment)
    
    return round(q_value, 3)