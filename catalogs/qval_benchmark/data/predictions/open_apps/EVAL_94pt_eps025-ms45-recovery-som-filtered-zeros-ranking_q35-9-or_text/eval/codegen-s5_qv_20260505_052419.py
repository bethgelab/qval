import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on state features and action effectiveness.
    Analyzes text representations to assess progress toward task completion.
    """
    # Base Q-value starts neutral
    q_value = 0.5
    
    # Extract bid numbers from state and next_state
    def extract_bids(text):
        return re.findall(r'bid(\d+)', text)
    
    current_bids = set(extract_bids(state))
    next_bids = set(extract_bids(next_state))
    
    # Check if action appears to be meaningful
    action_lower = action.lower()
    
    # Scoring based on state analysis
    score = 0.0
    
    # 1. Check for goal-related keywords in next_state
    goal_keywords = ['submit', 'save', 'add', 'create', 'send', 'confirm', 'done', 'complete', 'success', 'finish']
    next_state_lower = next_state.lower()
    
    for keyword in goal_keywords:
        if keyword in next_state_lower:
            score += 0.15
    
    # 2. Check for form completion indicators
    form_indicators = ['input', 'textarea', 'select', 'button', 'link', 'checkbox', 'radio']
    form_count_next = sum(1 for indicator in form_indicators if indicator in next_state_lower)
    form_count_current = sum(1 for indicator in form_indicators if indicator in state.lower())
    
    if form_count_next > form_count_current:
        score += 0.1
    
    # 3. Check for navigation progress
    nav_indicators = ['nav', 'menu', 'header', 'footer', 'sidebar', 'breadcrumb', 'home', 'back']
    nav_count_next = sum(1 for indicator in nav_indicators if indicator in next_state_lower)
    nav_count_current = sum(1 for indicator in nav_indicators if indicator in state.lower())
    
    if nav_count_next > nav_count_current:
        score += 0.05
    
    # 4. Check if action is meaningful (not noop)
    if 'noop' not in action_lower:
        score += 0.05
    
    # 5. Check for state change
    if next_state != state:
        score += 0.05
    
    # 6. Check for bid changes (element visibility changes)
    if len(next_bids) > len(current_bids):
        score += 0.05
    
    # 7. Check for bid removal (may indicate form submission or navigation)
    removed_bids = current_bids - next_bids
    if len(removed_bids) > 0:
        score += 0.03
    
    # 8. Check for specific task-related patterns in next_state
    task_patterns = ['calendar', 'event', 'message', 'todo', 'task', 'code', 'editor', 'map', 'location']
    for pattern in task_patterns:
        if pattern in next_state_lower:
            score += 0.08
            break
    
    # 9. Check for validation indicators
    validation_indicators = ['valid', 'error', 'warning', 'success', 'failed', 'accepted']
    for indicator in validation_indicators:
        if indicator in next_state_lower:
            score += 0.12
            break
    
    # 10. Check for button actions (often lead to completion)
    button_indicators = ['click', 'button', 'submit', 'save', 'confirm']
    if any(indicator in action_lower for indicator in button_indicators):
        score += 0.08
    
    # Normalize score to 0-1 range with smooth bounds
    q_value = min(1.0, max(0.0, 0.5 + score))
    
    return q_value