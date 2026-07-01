def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value starts at 0.5 (neutral)
    q_value = 0.5
    
    # Check for success indicators
    success_patterns = [
        r'successfully completed',
        r'task completed',
        r'you have won',
        r'thank you for playing',
        r'episode ended'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            q_value = 0.95
            return q_value
    
    # Check for error/failure indicators in next_state
    error_patterns = [
        r'nothing to take',
        r'nothing to put',
        r'there is no',
        r'cannot',
        r'invalid',
        r'error',
        r'fail',
        r'wrong',
        r'not available'
    ]
    
    error_count = 0
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            error_count += 1
    
    if error_count > 0:
        q_value -= 0.2 * error_count
    
    # Check for progress indicators
    progress_patterns = [
        r'pick up',
        r'put',
        r'place',
        r'clean',
        r'heat',
        r'cool',
        r'open',
        r'close',
        r'look at',
        r'go to'
    ]
    
    progress_count = 0
    for pattern in progress_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            progress_count += 1
    
    if progress_count > 0:
        q_value += 0.1 * min(progress_count, 3)
    
    # Check for object state changes (positive progress)
    state_change_patterns = [
        r'you put.*in',
        r'you place.*on',
        r'you clean',
        r'you heat',
        r'you cool',
        r'you pick up'
    ]
    
    state_change_count = 0
    for pattern in state_change_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            state_change_count += 1
    
    if state_change_count > 0:
        q_value += 0.15 * state_change_count
    
    # Check if action matches expected task flow (basic validation)
    action_words = action.lower().split()
    if len(action_words) >= 2:
        # Check if action has valid structure (verb + object/receptacle)
        valid_verbs = ['go', 'take', 'put', 'clean', 'heat', 'cool', 'open', 'close', 'look']
        if any(action_words[0] in verb for verb in valid_verbs):
            q_value += 0.05
    
    # Penalize if next_state is similar to state (no progress)
    if abs(len(state) - len(next_state)) < 50 and error_count == 0:
        q_value -= 0.1
    
    # Ensure Q-value stays in reasonable range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value