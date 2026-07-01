def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on terminal environment state analysis.
    Returns a float in [0, 1] representing expected success probability.
    """
    import re
    
    # Base Q-value starting from neutral
    q_value = 0.5
    
    # Analyze next_state for error indicators (negative signal)
    error_patterns = [
        r'\berror\b', r'\bfailed\b', r'\bfail\b', r'\bdenied\b',
        r'\bnot found\b', r'\bno such\b', r'\bpermission denied\b',
        r'\bsyntax error\b', r'\binvalid\b', r'\bcommand not found\b',
        r'\bfatal\b', r'\btraceback\b'
    ]
    
    next_state_lower = next_state.lower()
    error_count = sum(1 for pattern in error_patterns if re.search(pattern, next_state_lower))
    
    # Penalize for errors found in output
    q_value -= error_count * 0.15
    
    # Analyze for success/completion indicators (positive signal)
    success_patterns = [
        r'\bsuccess\b', r'\bcompleted\b', r'\bdone\b', r'\bpass\b',
        r'\bverified\b', r'\btest passed\b', r'\bsubmission\b',
        r'\bverified successfully\b', r'\btask complete\b'
    ]
    
    success_count = sum(1 for pattern in success_patterns if re.search(pattern, next_state_lower))
    q_value += success_count * 0.25
    
    # Check for productive file/directory operations
    productive_patterns = [
        r'\bcreated\b', r'\bwrote\b', r'\bsaved\b', r'\bmodified\b',
        r'\bupdated\b', r'\binstalled\b', r'\bcopied\b', r'\bextracted\b',
        r'\bcompiled\b', r'\bexecuted\b'
    ]
    
    productive_count = sum(1 for pattern in productive_patterns if re.search(pattern, next_state_lower))
    q_value += productive_count * 0.1
    
    # Check if action appears to be a valid command (not empty or trivial)
    action_stripped = action.strip()
    if action_stripped and len(action_stripped) > 1:
        # Penalize trivial actions
        trivial_actions = ['echo', 'pwd', 'ls', 'whoami', 'date', 'clear', 'exit']
        is_trivial = any(action_stripped.lower().startswith(cmd) for cmd in trivial_actions)
        if not is_trivial:
            q_value += 0.05
    
    # Check if next_state shows evidence of progress (file listings, outputs)
    progress_indicators = [
        r'\d+\s+bytes', r'\d+ files', r'total\s+\d+', r'\.py\b',
        r'\.txt\b', r'\.json\b', r'\.sh\b', r'\.md\b'
    ]
    progress_count = sum(1 for pattern in progress_indicators if re.search(pattern, next_state_lower))
    q_value += min(progress_count * 0.05, 0.15)
    
    # Penalize if state appears to be stuck or looping (repeated errors)
    if error_count >= 2 and success_count == 0:
        q_value -= 0.1
    
    # Bound Q-value to [0.0, 1.0] range
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)