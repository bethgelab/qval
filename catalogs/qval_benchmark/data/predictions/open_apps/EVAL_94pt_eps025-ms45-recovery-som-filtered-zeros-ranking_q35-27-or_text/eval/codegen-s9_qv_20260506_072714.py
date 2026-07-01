import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Keywords indicating goal achievement
    success_keywords = [
        'completed', 'success', 'done', 'created', 'added', 'sent',
        'saved', 'updated', 'finished', 'event', 'message', 'task',
        'calendar', 'todo', '✓', '✔', 'check', 'confirmed', 'posted'
    ]
    
    # Keywords indicating errors or failures
    failure_keywords = [
        'error', 'failed', 'invalid', 'not found', 'denied',
        'timeout', 'unauthorized', 'forbidden', '404', '500',
        'exception', 'crash', 'broken', 'missing', 'undefined'
    ]
    
    # Normalize text for analysis
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Base Q-value starting at 0.5 (neutral)
    q_value = 0.5
    
    # Check for success indicators in next_state
    has_success = any(kw in next_state_lower for kw in success_keywords)
    has_failure = any(kw in next_state_lower for kw in failure_keywords)
    
    # Adjust based on goal achievement status
    if has_success and not has_failure:
        q_value = 0.95
    elif has_failure:
        q_value = 0.1
    elif has_success:
        q_value = 0.8
    
    # Evaluate action type productivity
    if 'click' in action_lower or 'fill' in action_lower or 'press' in action_lower:
        q_value += 0.1
    elif 'noop' in action_lower:
        q_value -= 0.15
    elif 'scroll' in action_lower:
        q_value -= 0.05
    
    # Check for bid interactions (BrowserGym uses bid numbers)
    bid_pattern = r"bid\s*['\"]?(\d+)['\"]?"
    action_bids = re.findall(bid_pattern, action)
    next_bids = re.findall(bid_pattern, next_state)
    
    # If action involves a bid that appears in next_state, it's likely productive
    if action_bids and any(bid in next_bids for bid in action_bids):
        q_value += 0.05
    
    # Check for form-related progress indicators
    form_progress = ['filled', 'entered', 'input', 'value=', 'text', 'content']
    has_form_progress = any(kw in next_state_lower for kw in form_progress)
    if has_form_progress:
        q_value += 0.05
    
    # Check for navigation progress (page changes, URL updates)
    nav_progress = ['href', 'url', 'link', 'navigate', 'page', 'route']
    has_nav_progress = any(kw in next_state_lower for kw in nav_progress)
    if has_nav_progress and not has_failure:
        q_value += 0.03
    
    # Clamp Q-value to valid range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value