import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on state transition analysis.
    
    Returns a float in [0, 1] representing estimated expected return.
    """
    # Start with neutral estimate
    q_value = 0.5
    
    # Analyze action for task relevance
    action_lower = action.lower() if action else ""
    
    # Actions that typically move toward goal completion
    positive_action_patterns = [
        'click', 'fill', 'press', 'submit', 'save', 'create', 'add', 
        'send', 'delete', 'edit', 'open', 'navigate', 'type', 'enter',
        'scroll', 'select', 'focus'
    ]
    
    # Check if action involves specific element interaction
    has_element_target = any(pattern in action_lower for pattern in ['bid', 'button', 'link', 'input', 'textarea', 'form'])
    
    # Check for meaningful action keywords
    has_meaningful_action = any(pattern in action_lower for pattern in positive_action_patterns)
    
    # Analyze state progression
    # Look for bid numbers (element identifiers)
    state_bids = re.findall(r'bid\s*\d+', state)
    next_bids = re.findall(r'bid\s*\d+', next_state)
    
    # Check for new elements appearing (page navigation or content change)
    if len(next_bids) > len(state_bids):
        q_value += 0.15
    
    # Check for bid disappearance (element removed or page changed)
    if len(next_bids) < len(state_bids):
        q_value -= 0.1
    
    # Look for goal completion indicators in next_state
    success_keywords = ['complete', 'success', 'done', 'saved', 'created', 'added', 'sent', 'received', 'confirmed', 'ok', 'ready']
    if any(kw in next_state.lower() for kw in success_keywords):
        q_value += 0.25
    
    # Look for error indicators in next_state
    error_keywords = ['error', 'failed', 'invalid', 'missing', 'not found', 'unavailable', 'unable', 'problem', 'issue']
    if any(kw in next_state.lower() for kw in error_keywords):
        q_value -= 0.3
    
    # Check if action involves meaningful interaction with state
    if has_meaningful_action and has_element_target:
        q_value += 0.05
    
    # Ensure Q-value stays in reasonable range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value