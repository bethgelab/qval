def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Normalize strings for analysis
    next_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Completion indicators - signals that task may be done
    completion_patterns = [
        'success', 'completed', 'saved', 'submitted', 'created',
        'added', 'sent', 'confirmation', 'done', 'finished'
    ]
    
    # Count completion signals in next state
    completion_count = sum(1 for pattern in completion_patterns if pattern in next_lower)
    
    # If multiple completion signals present, high Q-value
    if completion_count >= 2:
        return 0.95
    
    # Single completion signal suggests near-completion
    if completion_count == 1:
        return 0.75
    
    # Base Q-value for meaningful action
    q_value = 0.5
    
    # Penalize no-op actions - they don't progress toward goal
    if 'noop' in action_lower:
        q_value -= 0.25
    
    # Reward direct interaction actions
    if 'fill' in action_lower:
        q_value += 0.15
    if 'click' in action_lower:
        q_value += 0.10
    
    # Check if action targets specific elements (contains bid numbers)
    if re.search(r'\d+', action_lower):
        q_value += 0.05
    
    # Penalize scroll-only actions (indirect progress)
    if 'scroll' in action_lower and 'click' not in action_lower and 'fill' not in action_lower:
        q_value -= 0.10
    
    # Check for meaningful state change (significant difference in content)
    state_change = abs(len(next_state) - len(state))
    if state_change > 200:
        q_value += 0.10
    elif state_change > 50:
        q_value += 0.05
    
    # Penalize if state barely changed (ineffective action)
    if state_change < 10 and 'noop' not in action_lower:
        q_value -= 0.05
    
    # Check for progress indicators appearing in next state
    progress_patterns = ['step', 'page', 'view', 'list', 'item', 'event', 'message']
    progress_gain = sum(1 for p in progress_patterns if p in next_lower and p not in state_lower)
    q_value += progress_gain * 0.03
    
    # Check for error indicators (negative signal)
    error_patterns = ['error', 'fail', 'invalid', 'wrong', 'not found']
    error_count = sum(1 for p in error_patterns if p in next_lower)
    if error_count > 0:
        q_value -= 0.20
    
    # Clamp to valid Q-value range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value