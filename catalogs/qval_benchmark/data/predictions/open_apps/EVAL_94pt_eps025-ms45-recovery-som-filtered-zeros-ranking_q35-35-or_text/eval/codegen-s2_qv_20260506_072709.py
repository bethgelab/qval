import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value initialization
    q_value = 0.0
    
    # Check for goal completion indicators in next_state
    goal_keywords = ['completed', 'success', 'done', 'submitted', 'sent', 'saved', 
                     'added', 'created', 'confirmed', 'confirmed', 'event', 'message',
                     'task', 'item', 'entry', 'scheduled', 'updated']
    
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Check if goal appears to be achieved in next_state
    goal_achieved = False
    for keyword in goal_keywords:
        if keyword in next_state_lower:
            goal_achieved = True
            break
    
    # Check for explicit success indicators
    success_patterns = [
        r'success', r'completed', r'done', r'saved', r'submitted',
        r'sent', r'added', r'created', r'confirmed', r'updated'
    ]
    for pattern in success_patterns:
        if re.search(pattern, next_state_lower):
            goal_achieved = True
            break
    
    # Check for error indicators (negative signal)
    error_patterns = [
        r'error', r'failed', r'unable', r'invalid', r'not found',
        r'denied', r'rejected', r'missing', r'require'
    ]
    has_error = False
    for pattern in error_patterns:
        if re.search(pattern, next_state_lower):
            has_error = True
            break
    
    # Evaluate action type appropriateness
    action_score = 0.5  # Neutral baseline
    
    # Good actions that typically progress toward goals
    productive_actions = ['click', 'fill', 'press', 'submit', 'navigate']
    for act in productive_actions:
        if act in action_lower:
            action_score = max(action_score, 0.6)
            break
    
    # Check if action seems to complete a form or task
    if 'submit' in action_lower or 'confirm' in action_lower:
        action_score = max(action_score, 0.8)
    
    # Check if we're clicking on goal-related elements
    goal_element_patterns = [
        r'submit', r'send', r'save', r'add', r'create', r'new',
        r'confirm', r'complete', r'done', r'schedule', r'set'
    ]
    for pattern in goal_element_patterns:
        if re.search(pattern, action_lower):
            action_score = max(action_score, 0.7)
            break
    
    # Measure progress between state and next_state
    progress_score = 0.5
    
    # Look for changes that indicate forward progress
    state_words = set(state_lower.split())
    next_state_words = set(next_state_lower.split())
    
    # New content appeared (positive signal)
    new_content = next_state_words - state_words
    if len(new_content) > 5:
        progress_score = 0.7
    
    # Check for navigation depth changes (moving closer to goal)
    breadcrumb_patterns = [r'home', r'back', r'previous', r'forward', r'next']
    in_state = any(re.search(p, state_lower) for p in breadcrumb_patterns)
    in_next = any(re.search(p, next_state_lower) for p in breadcrumb_patterns)
    
    if in_next and not in_state:
        progress_score = max(progress_score, 0.6)
    
    # Form completion indicators
    form_indicators = ['form', 'input', 'field', 'text', 'checkbox', 'radio', 'select']
    if any(ind in next_state_lower for ind in form_indicators):
        # Check if we're filling forms (progress)
        if 'fill' in action_lower or 'input' in action_lower:
            progress_score = max(progress_score, 0.7)
    
    # Calculate final Q-value based on components
    if goal_achieved:
        q_value = 1.0
    elif has_error:
        q_value = 0.0
    else:
        # Combine scores with weighted average
        q_value = (
            0.4 * action_score +
            0.3 * progress_score +
            0.3 * 0.5  # Base progress baseline
        )
    
    # Apply slight bonus for actions that seem to complete tasks
    completion_indicators = ['click', 'submit', 'confirm', 'done', 'complete']
    if any(ind in action_lower for ind in completion_indicators):
        q_value = min(q_value + 0.1, 1.0)
    
    # Ensure output is in valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value