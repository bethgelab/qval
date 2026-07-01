def signal_function(state: str, action: str, next_state: str) -> float:
    # Initialize Q-value estimate
    q_value = 0.0
    
    # Positive indicators in current state
    positive_patterns = ['success', 'done', 'complete', 'created', 'wrote', 'saved', 
                        'output:', 'result:', 'generated', 'processed', 'uploaded',
                        'installed', 'configured', 'enabled', 'started', 'running']
    # Negative indicators in current state
    negative_patterns = ['error', 'fail', 'failed', 'exception', 'traceback', 
                        'fatal', 'invalid', 'missing', 'permission denied', 'refused',
                        'timeout', 'interrupted', 'broken', 'corrupted', 'overflow']
    
    # Count positive and negative signals in state
    state_pos = sum(1 for p in positive_patterns if p.lower() in state.lower())
    state_neg = sum(1 for p in negative_patterns if p.lower() in state.lower())
    
    # Count positive and negative signals in next state
    next_pos = sum(1 for p in positive_patterns if p.lower() in next_state.lower())
    next_neg = sum(1 for p in negative_patterns if p.lower() in next_state.lower())
    
    # Count positive and negative signals in action
    action_pos = sum(1 for p in positive_patterns if p.lower() in action.lower())
    action_neg = sum(1 for p in negative_patterns if p.lower() in action.lower())
    
    # Calculate state quality score
    state_score = state_pos - state_neg
    
    # Calculate next state quality score
    next_state_score = next_pos - next_neg
    
    # Calculate action quality score
    action_score = action_pos - action_neg
    
    # Improvement score: how much better is next state compared to current
    improvement = next_state_score - state_score
    
    # Base Q-value on current state quality
    base_value = state_score * 0.3
    
    # Add expected improvement
    improvement_value = improvement * 0.4
    
    # Factor in action quality
    action_value = action_score * 0.2
    
    # Factor in next state absolute quality
    next_value = next_state_score * 0.1
    
    # Combine all factors
    q_value = base_value + improvement_value + action_value + next_value
    
    # Clamp to reasonable range [-1, 1]
    q_value = max(-1.0, min(1.0, q_value))
    
    # If next state appears to show clear completion, boost value
    completion_indicators = ['task completed', 'success', 'exit code 0', 'return 0', 
                            'done', 'finished', 'all tests passed']
    if any(c in next_state.lower() for c in completion_indicators):
        q_value = min(1.0, q_value + 0.3)
    
    # If next state shows clear failure, reduce value
    failure_indicators = ['error', 'fail', 'exit code 1', 'return 1', 
                         'exception', 'fatal', 'permission denied']
    if any(f in next_state.lower() for f in failure_indicators):
        q_value = max(-1.0, q_value - 0.4)
    
    return q_value