def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Keywords indicating task completion or strong success signals
    completion_keywords = [
        'success', 'completed', 'done', 'finished', 'passed', 'verified',
        'valid', 'correct', 'ok', 'true', '100%', 'complete', 'ready',
        'final', 'result', 'answer', 'solution', 'output', 'generated',
        'created', 'wrote', 'saved', 'exported', 'built', 'compiled',
        'test.*pass', 'pass.*test', 'verification.*success'
    ]
    
    # Keywords indicating errors or failure states
    error_keywords = [
        'error', 'fail', 'failed', 'exception', 'traceback', 'warning',
        'not found', 'permission denied', 'invalid', 'incorrect', 'wrong',
        'missing', 'undefined', 'none', 'null', 'false', 'timeout',
        'interrupted', 'killed', 'segmentation', 'core dump', 'denied',
        'permission', 'access denied', 'refused', 'connection refused'
    ]
    
    # Keywords indicating intermediate progress
    progress_keywords = [
        'processing', 'computing', 'calculating', 'building', 'creating',
        'generating', 'writing', 'saving', 'exporting', 'downloading',
        'uploading', 'fetching', 'reading', 'parsing', 'analyzing',
        'running', 'executing', 'working', 'in progress', 'step'
    ]
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Count keyword occurrences in next_state
    completion_score = sum(1 for kw in completion_keywords if kw in next_state_lower)
    error_score = sum(1 for kw in error_keywords if kw in next_state_lower)
    progress_score = sum(1 for kw in progress_keywords if kw in next_state_lower)
    
    # Count in current state for progress comparison
    state_completion = sum(1 for kw in completion_keywords if kw in state_lower)
    state_error = sum(1 for kw in error_keywords if kw in state_lower)
    
    # Check for strong completion signals
    strong_completion = any(
        kw in next_state_lower for kw in 
        ['done', 'completed', 'finished', 'success', 'passed', 'verified', 'complete']
    )
    
    # Check for strong error signals
    strong_error = any(
        kw in next_state_lower for kw in
        ['error', 'fail', 'failed', 'exception', 'not found', 'permission denied']
    )
    
    # Calculate progress delta (improvement from state to next_state)
    progress_delta = (completion_score - state_completion) - (error_score - state_error)
    
    # Base Q-value estimate (neutral starting point)
    q_value = 0.5
    
    # Strong completion boost - likely near goal
    if strong_completion:
        q_value = 0.9
    
    # Strong error penalty - likely in bad state
    if strong_error:
        q_value = 0.2
    
    # Adjust based on keyword scores
    q_value += completion_score * 0.06
    q_value -= error_score * 0.1
    q_value += progress_score * 0.03
    
    # Adjust based on progress delta
    q_value += progress_delta * 0.04
    
    # Bonus for meaningful actions (not just navigation commands)
    if len(action.strip()) > 2 and action_lower not in ['cd', 'pwd', 'ls', 'ls -la', 'ls -l']:
        q_value += 0.02
    
    # Check for file creation/modification indicators (common in terminal tasks)
    file_indicators = ['created', 'wrote', 'saved', 'file', 'directory', 'mkdir', 'touch', 'cp', 'mv']
    file_score = sum(1 for kw in file_indicators if kw in next_state_lower)
    q_value += file_score * 0.02
    
    # Check for command success patterns (exit code 0, etc.)
    if re.search(r'exit.*0|return.*0|code.*0', next_state_lower):
        q_value += 0.05
    
    # Penalty for very long output (might indicate issues or verbosity)
    if len(next_state) > 5000:
        q_value -= 0.05
    
    # Clamp to valid Q-value range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value