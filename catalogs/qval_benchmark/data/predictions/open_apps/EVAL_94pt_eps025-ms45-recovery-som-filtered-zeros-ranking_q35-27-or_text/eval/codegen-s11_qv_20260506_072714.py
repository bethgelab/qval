import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Maximum episode steps
    MAX_STEPS = 45
    
    # Base discount factor per step
    DISCOUNT = 0.95
    
    # Check for goal completion indicators in next_state
    goal_keywords = [
        'success', 'completed', 'saved', 'created', 'added', 
        'sent', 'updated', 'done', 'finished', 'event created',
        'message sent', 'task added', 'calendar event', 'todo item'
    ]
    next_state_lower = next_state.lower()
    
    goal_achieved = any(keyword in next_state_lower for keyword in goal_keywords)
    
    # If goal achieved, return high value
    if goal_achieved:
        return 1.0
    
    # Check for error/failure indicators
    error_keywords = [
        'error', 'failed', 'invalid', 'cannot', 'unable', 
        'denied', 'permission', 'not found', '404', '500'
    ]
    has_error = any(keyword in next_state_lower for keyword in error_keywords)
    
    if has_error:
        return -0.1
    
    # Analyze action type
    action_lower = action.lower()
    
    # Action value scoring
    action_value = 0.0
    
    # Productive actions
    if 'click' in action_lower:
        action_value += 0.3
    elif 'fill' in action_lower:
        action_value += 0.4
    elif 'press' in action_lower:
        action_value += 0.25
    elif 'scroll' in action_lower:
        action_value += 0.05  # Less valuable but sometimes necessary
    elif 'noop' in action_lower:
        action_value -= 0.2  # Wasteful
    else:
        action_value += 0.1  # Unknown action, slight positive bias
    
    # Check for meaningful state changes
    state_lower = state.lower()
    
    # Measure state difference (simple heuristic)
    state_words = set(re.findall(r'\w+', state_lower))
    next_state_words = set(re.findall(r'\w+', next_state_lower))
    
    new_words = next_state_words - state_words
    removed_words = state_words - next_state_words
    
    # Progress indicators
    progress_indicators = [
        'new', 'added', 'created', 'updated', 'changed', 
        'selected', 'opened', 'loaded', 'displayed'
    ]
    
    progress_score = 0.0
    for word in new_words:
        if word in progress_indicators:
            progress_score += 0.1
        if len(word) > 3:  # Likely meaningful content
            progress_score += 0.02
    
    # Penalize if state barely changed (possible stuck state)
    if len(new_words) < 3 and len(removed_words) < 3:
        progress_score -= 0.1
    
    # Check for navigation/progress cues
    nav_indicators = [
        'page', 'section', 'view', 'menu', 'header', 
        'button', 'form', 'input', 'field', 'link'
    ]
    
    nav_score = 0.0
    for word in new_words:
        if word in nav_indicators:
            nav_score += 0.05
    
    # Estimate remaining steps based on state complexity
    # More content = potentially more steps needed
    state_length = len(state)
    estimated_remaining = max(5, 20 - (state_length / 1000))
    
    # Calculate step efficiency factor
    # Closer to goal = higher value
    efficiency_factor = max(0.1, 1.0 - (estimated_remaining / MAX_STEPS))
    
    # Combine scores
    base_q = action_value + progress_score + nav_score
    
    # Apply efficiency discount
    q_value = base_q * efficiency_factor
    
    # Cap the value
    q_value = max(-0.5, min(0.9, q_value))
    
    return q_value