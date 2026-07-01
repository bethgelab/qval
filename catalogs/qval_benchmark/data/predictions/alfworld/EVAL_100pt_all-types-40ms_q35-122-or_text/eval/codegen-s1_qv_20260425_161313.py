def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Normalize text for analysis
    state_text = state.lower()
    next_text = next_state.lower()
    action_text = action.lower()
    
    # Check for terminal success state
    success_indicators = ['success', 'completed', 'task complete', 'you have completed']
    if any(ind in next_text for ind in success_indicators):
        return 1.0
    
    # Check for terminal failure state
    failure_indicators = ['failed', 'error', 'max steps', 'exceeded']
    if any(ind in next_text for ind in failure_indicators):
        return 0.0
    
    # Start with baseline Q-value
    q_value = 0.5
    
    # Score based on action productivity
    productive_actions = ['go to', 'walk to', 'move to', 'take', 'pick up', 'put', 
                          'place', 'clean', 'heat', 'cool', 'open', 'close', 'examine']
    if any(act in action_text for act in productive_actions):
        q_value += 0.15
    
    # Score based on successful outcome
    success_outcomes = ['you pick', 'you take', 'you put', 'you place', 'you clean',
                        'you heat', 'you cool', 'you open', 'you close', 'successfully']
    if any(out in next_text for out in success_outcomes):
        q_value += 0.2
    
    # Score based on location/progress indicators
    progress_indicators = ['in the', 'on the', 'at the', 'near the', 'next to the']
    if any(ind in next_text for ind in progress_indicators):
        q_value += 0.1
    
    # Penalize failed actions
    failure_outcomes = ['cannot', 'not found', 'nothing', 'empty', 'already', 
                        'does not', 'no such', 'invalid', 'not here', 'nothing to']
    if any(out in next_text for out in failure_outcomes):
        q_value -= 0.2
    
    # Penalize navigation without progress
    if action_text.startswith('go to') or action_text.startswith('walk to'):
        if 'nothing' in next_text or 'empty' in next_text:
            q_value -= 0.1
    
    # Adjust for efficiency (fewer steps = better)
    # Estimate steps used by counting state transitions
    step_penalty = 0.0
    if len(state_text) > len(next_text):
        step_penalty = 0.05  # State got simpler, likely progress
    q_value += step_penalty
    
    # Ensure Q-value stays in valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value