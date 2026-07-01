def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    q_value = 0.0
    
    next_lower = next_state.lower()
    action_lower = action.lower()
    
    # Penalize error/invalid actions
    error_patterns = ['cannot', 'already', 'not', 'empty', 'invalid', 'does not',
                      'is not', 'nothing', 'no', 'unable', 'fail', 'error']
    has_error = any(pattern in next_lower for pattern in error_patterns)
    if has_error:
        q_value -= 0.6
    
    # Reward successful task completion indicators
    success_patterns = ['done', 'completed', 'success', 'goal', 'finished',
                        'task complete', 'solved']
    has_success = any(pattern in next_lower for pattern in success_patterns)
    if has_success:
        q_value += 1.5
    
    # Reward meaningful object interactions
    interaction_patterns = ['picked up', 'taken', 'added to inventory',
                            'in your inventory', 'put', 'placed', 'moved',
                            'opened', 'closed', 'turned on', 'turned off',
                            'cleaned', 'washed', 'rubbed', 'heated', 'cooled']
    interaction_count = sum(1 for pattern in interaction_patterns if pattern in next_lower)
    q_value += interaction_count * 0.3
    
    # Reward navigation progress
    nav_patterns = ['went', 'walked', 'navigated', 'room', 'entered', 'exited']
    nav_count = sum(1 for pattern in nav_patterns if pattern in next_lower)
    q_value += nav_count * 0.2
    
    # Reward action relevance
    action_keywords = ['go', 'move', 'take', 'put', 'open', 'close', 'clean',
                       'turn', 'use', 'examine', 'inventory', 'pick', 'drop',
                       'heat', 'cool', 'rub', 'wash', 'fill', 'empty']
    action_relevant = any(keyword in action_lower for keyword in action_keywords)
    if action_relevant:
        q_value += 0.25
    
    # Penalize no-op or redundant actions
    if 'nothing' in next_lower or 'same' in next_lower or 'unchanged' in next_lower:
        q_value -= 0.3
    
    # Bonus for finding items (often key to task completion)
    if 'found' in next_lower or 'see' in next_lower or 'located' in next_lower:
        q_value += 0.4
    
    # Cap the Q-value to reasonable bounds
    q_value = max(-1.0, min(2.0, q_value))
    
    return q_value