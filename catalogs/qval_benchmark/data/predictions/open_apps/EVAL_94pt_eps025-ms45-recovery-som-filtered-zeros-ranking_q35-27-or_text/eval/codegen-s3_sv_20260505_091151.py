def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Goal completion indicators - high value
    completion_terms = ['success', 'completed', 'done', 'created', 'added', 'saved', 'sent', 'updated', 'finished', 'complete', 'task complete', 'goal achieved', 'successfully']
    if any(term in state_lower for term in completion_terms):
        return 0.95
    
    # Error/failure indicators - low value
    error_terms = ['error', 'failed', 'invalid', 'denied', 'rejected', 'warning', 'cannot', 'unable', 'not found', 'failed to', 'exception', 'crash']
    if any(term in state_lower for term in error_terms):
        return 0.1
    
    # Count interactive bid elements (available actions)
    bid_count = len(re.findall(r'bid["\']?\s*[:=]\s*["\']?\d+', state))
    
    # Count form/action elements
    action_elements = state_lower.count('button') + state_lower.count('input') + state_lower.count('link') + state_lower.count('form')
    
    # Check for content that suggests task progress
    progress_indicators = state_lower.count('value') + state_lower.count('content') + state_lower.count('text') + state_lower.count('item')
    
    # Calculate base value from available actions and elements
    action_score = min(bid_count * 0.02, 0.35)
    element_score = min(action_elements * 0.03, 0.3)
    content_score = min(progress_indicators * 0.02, 0.2)
    
    base_value = 0.3 + action_score + element_score + content_score
    
    # Slight penalty for very sparse states (few elements)
    if bid_count < 3 and action_elements < 5:
        base_value *= 0.8
    
    return max(0.0, min(1.0, base_value))