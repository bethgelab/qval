import re
import json

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on state, action, and next_state analysis.
    
    Higher values indicate more favorable states/actions that lead to goal achievement.
    """
    q_value = 0.0
    
    # Check if goal was achieved in next state
    goal_indicators = ['goal', 'success', 'completed', 'done', 'achieved', 
                       'task complete', 'success message', 'verified', 'saved', 
                       'added', 'created', 'sent', 'delivered']
    if any(ind in next_state.lower() for ind in goal_indicators):
        q_value = 1.0
        return q_value
    
    # Check for error/warning indicators
    error_indicators = ['error', 'failed', 'invalid', 'not found', 
                        'unavailable', 'cannot', 'warning', 'alert', 
                        'blocked', 'denied', 'refused', 'timeout']
    if any(ind in next_state.lower() for ind in error_indicators):
        q_value = -0.5
    
    # Check if action led to meaningful state change
    if state == next_state:
        q_value = -0.3  # No progress made
    
    # Check for navigation progress
    page_indicators = ['page', 'view', 'screen', 'section', 'tab', 'url', 'address']
    state_pages = [p for p in page_indicators if p in state.lower()]
    next_pages = [p for p in page_indicators if p in next_state.lower()]
    
    if next_pages and not state_pages:
        q_value += 0.2  # New page loaded
    
    if state_pages and not next_pages:
        q_value -= 0.2  # Page unloaded (possibly bad)
    
    # Check for form completion indicators
    form_indicators = ['filled', 'submitted', 'entered', 'value', 'input', 
                       'checkbox', 'radio', 'select', 'dropdown', 'button']
    state_form_count = len(re.findall(rf'\b{re.escape(" ".join(form_indicators))}\b', state.lower()))
    next_form_count = len(re.findall(rf'\b{re.escape(" ".join(form_indicators))}\b', next_state.lower()))
    
    if next_form_count > state_form_count:
        q_value += 0.1
    
    # Check for backward navigation (going back to previous state)
    if state in next_state and len(next_state) > len(state):
        q_value -= 0.1
    
    # Check action type
    if 'click' in action.lower():
        q_value += 0.05  # Clicks are common navigation actions
    elif 'fill' in action.lower():
        q_value += 0.1  # Filling forms is progress
    elif 'press' in action.lower():
        q_value += 0.05  # Keyboard navigation
    elif 'scroll' in action.lower():
        q_value += 0.02  # Scrolling is neutral to slightly positive
    elif 'noop' in action.lower():
        q_value -= 0.1  # No operation is neutral to negative
    
    # Check for positive state changes
    positive_indicators = ['new', 'updated', 'modified', 'changed', 'added', 
                           'opened', 'loaded', 'displayed', 'visible', 'enabled']
    if any(ind in next_state.lower() and ind not in state.lower() 
           for ind in positive_indicators):
        q_value += 0.05
    
    # Check for negative state changes
    negative_indicators = ['disabled', 'hidden', 'removed', 'deleted', 
                           'closed', 'unavailable', 'missing', 'empty', 'blank']
    if any(ind in next_state.lower() and ind not in state.lower() 
           for ind in negative_indicators):
        q_value -= 0.05
    
    # Check for bid/tag changes (indicating element interactions)
    bid_pattern = r'\bid=(\d+)'
    state_bids = re.findall(bid_pattern, state)
    next_bids = re.findall(bid_pattern, next_state)
    
    if len(next_bids) > len(state_bids):
        q_value += 0.02  # More interactive elements visible
    
    # Check for step efficiency (shorter text might indicate progress)
    if len(next_state) < len(state) and len(next_state) > len(state) * 0.5:
        q_value += 0.05  # Concise state might indicate progress
    
    # Clamp value
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value