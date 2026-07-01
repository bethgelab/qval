import re
import json
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    # Heuristic scoring based on state analysis without recursion or lookahead
    
    # Define goal-related keywords that indicate progress
    goal_keywords = ['goal', 'task', 'complete', 'success', 'done', 'saved', 'created', 
                     'added', 'event', 'message', 'calendar', 'todo', 'map', 'code',
                     'submit', 'finish', 'target', 'expected', 'achieved']
    
    # Define error/warning keywords that indicate bad states
    error_keywords = ['error', 'fail', 'invalid', 'missing', 'not found', 'unable', 
                      'blocked', 'timeout', 'exception', 'alert', 'warning']
    
    # Define action-related keywords for progress indicators
    progress_indicators = ['bid', 'click', 'fill', 'button', 'link', 'form', 'input', 
                          'field', 'page', 'section', 'header', 'footer', 'nav']
    
    # Score current state
    state_score = 0.0
    state_keywords = [kw.lower() for kw in goal_keywords]
    error_keywords_list = [kw.lower() for kw in error_keywords]
    
    state_lower = state.lower()
    
    # Count goal-related words
    goal_count = sum(1 for kw in state_keywords if kw in state_lower)
    state_score += min(goal_count * 0.1, 0.5)
    
    # Check for error indicators
    error_count = sum(1 for kw in error_keywords_list if kw in state_lower)
    state_score -= min(error_count * 0.3, 0.5)
    
    # Count interactive elements (bid tags)
    bid_matches = re.findall(r'\bid\d+', state)
    state_score += min(len(bid_matches) * 0.02, 0.3)
    
    # Score next state
    next_state_keywords = [kw.lower() for kw in goal_keywords]
    next_state_lower = next_state.lower()
    
    next_goal_count = sum(1 for kw in next_state_keywords if kw in next_state_lower)
    next_error_count = sum(1 for kw in error_keywords_list if kw in next_state_lower)
    
    next_state_score = min(goal_count * 0.1, 0.5)
    next_state_score -= min(next_error_count * 0.3, 0.5)
    
    # Check for improvement
    improvement = next_state_score - state_score
    improvement_score = min(max(improvement * 2, 0), 0.5)
    
    # Analyze action type
    action_lower = action.lower()
    action_score = 0.0
    
    # Click actions are common for navigation
    if 'click' in action_lower:
        action_score += 0.2
    elif 'fill' in action_lower:
        action_score += 0.3
    elif 'press' in action_lower:
        action_score += 0.15
    elif 'scroll' in action_lower:
        action_score += 0.1
    
    # Check if action seems appropriate (contains bid reference)
    if 'bid' in action_lower:
        action_score += 0.2
    
    # Check if next state appears more progressed than current
    if len(next_state) > len(state) * 0.8:
        action_score += 0.1
    
    # Combine scores
    base_q = 0.3 + state_score + improvement_score + action_score
    
    # Normalize to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, base_q))
    
    # Bonus if goal keywords appear more in next state
    if next_goal_count > goal_count:
        q_value = min(q_value + 0.2, 1.0)
    
    # Penalty if error keywords appear more in next state
    if next_error_count > error_count:
        q_value = max(q_value - 0.2, 0.0)
    
    return round(q_value, 4)