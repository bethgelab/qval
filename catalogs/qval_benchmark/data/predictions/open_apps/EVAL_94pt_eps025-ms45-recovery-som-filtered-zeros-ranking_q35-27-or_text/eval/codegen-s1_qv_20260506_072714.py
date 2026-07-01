import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for goal completion indicators in next_state
    success_indicators = ['completed', 'success', 'saved', 'added', 'created', 'sent', 
                          'updated', 'done', 'finished', '✓', '✔', 'event', 'message']
    next_state_lower = next_state.lower()
    goal_achieved = any(ind in next_state_lower for ind in success_indicators)
    
    if goal_achieved:
        return 1.0
    
    # Evaluate action quality
    action_lower = action.lower()
    action_score = 0.0
    
    # Productive actions
    if action_lower.startswith('click(') or action_lower.startswith('fill('):
        action_score = 0.6
    elif action_lower.startswith('press('):
        action_score = 0.5
    elif action_lower.startswith('noop('):
        action_score = 0.1
    elif action_lower.startswith('scroll('):
        action_score = 0.2
    else:
        action_score = 0.3
    
    # Detect progress by comparing state and next_state
    state_words = set(re.findall(r'\w+', state.lower()))
    next_words = set(re.findall(r'\w+', next_state.lower()))
    
    # New content in next_state suggests progress
    new_content = len(next_words - state_words)
    content_change = len(state_words.symmetric_difference(next_words))
    
    # Progress score based on state changes
    progress_score = 0.0
    if new_content > 5:
        progress_score = 0.3
    elif new_content > 2:
        progress_score = 0.2
    elif content_change > 10:
        progress_score = 0.25
    
    # Check for form field indicators (suggests we're in a productive state)
    form_indicators = ['input', 'text', 'email', 'phone', 'password', 'submit', 
                       'button', 'form', 'label', 'placeholder']
    has_form = any(ind in next_state_lower for ind in form_indicators)
    form_bonus = 0.1 if has_form else 0.0
    
    # Check for navigation elements (suggests we can reach goal)
    nav_indicators = ['link', 'href', 'nav', 'menu', 'tab', 'section']
    has_nav = any(ind in next_state_lower for ind in nav_indicators)
    nav_bonus = 0.1 if has_nav else 0.0
    
    # Detect if we're stuck (minimal change, unproductive action)
    if content_change < 3 and action_score < 0.3:
        stuck_penalty = 0.3
    else:
        stuck_penalty = 0.0
    
    # Estimate remaining steps based on state complexity
    # More content generally means we're further along
    content_ratio = min(1.0, len(next_words) / 500.0)
    step_discount = 0.95 ** (20 - int(content_ratio * 25))  # Assume ~20 steps optimal
    
    # Combine all factors
    base_value = action_score * 0.4 + progress_score * 0.3 + form_bonus + nav_bonus
    q_value = base_value * step_discount - stuck_penalty
    
    # Clamp to valid range
    return max(0.0, min(1.0, q_value))