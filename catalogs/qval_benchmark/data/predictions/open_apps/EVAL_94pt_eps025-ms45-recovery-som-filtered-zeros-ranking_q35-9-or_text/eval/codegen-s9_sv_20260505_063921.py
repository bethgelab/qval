import re
import math
from collections import Counter

def signal_function(state: str) -> float:
    """
    Estimate the state-value V(s) for the OpenApps environment.
    
    The value is based on heuristic analysis of the state text representation,
    considering goal proximity, available actions, and error indicators.
    """
    if not state or len(state) < 10:
        return 0.0
    
    # Check for goal completion indicators
    goal_indicators = ['goal achieved', 'task complete', 'success', 'saved', 'added', 
                       'created', 'sent', 'event added', 'message sent', 'done', 
                       'completed', '✓', 'SUCCESS', 'Goal met']
    has_goal = any(ind in state.lower() for ind in goal_indicators)
    
    # Check for error indicators
    error_indicators = ['error', 'failed', 'invalid', 'cannot', 'unable', 
                        'not found', 'missing', 'invalid', '×', 'ERROR', 'fail']
    has_error = any(ind in state.lower() for ind in error_indicators)
    
    # Count interactive elements (potential actions)
    bid_pattern = r'\bid\d+'
    bids = re.findall(bid_pattern, state)
    num_actions = len(bids)
    
    # Count form elements
    input_pattern = r'\bid\d+.*?(input|text|button|select|textarea)'
    form_elements = len(re.findall(input_pattern, state, re.IGNORECASE))
    
    # Check for navigation elements
    nav_pattern = r'(link|button|nav|menu|back|next|forward)'
    nav_elements = len(re.findall(nav_pattern, state, re.IGNORECASE))
    
    # Check for page context indicators
    page_indicators = ['todo', 'calendar', 'messenger', 'maps', 'code', 
                       'editor', 'app', 'page', 'view', 'screen']
    has_page_context = any(ind in state.lower() for ind in page_indicators)
    
    # Calculate base value from goal status
    if has_goal:
        return 0.95
    
    # Penalize if errors detected
    if has_error:
        return 0.1
    
    # Calculate action richness score
    action_score = min(num_actions / 10.0, 1.0)
    form_score = min(form_elements / 5.0, 1.0)
    nav_score = min(nav_elements / 3.0, 1.0)
    
    # Calculate overall state quality
    quality = (action_score + form_score + nav_score) / 3.0
    
    # Adjust based on page context presence
    if has_page_context:
        quality *= 1.2
    
    # Ensure value is in [0, 1] range
    value = max(0.0, min(1.0, quality))
    
    # Add small bonus for states with reasonable action availability
    if 2 <= num_actions <= 10:
        value += 0.05
    
    return round(value, 4)