def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value
    q_value = 0.5
    
    # Normalize text for analysis
    next_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Success indicators (increase Q-value)
    success_patterns = [
        r'\bsuccess\b', r'\bcompleted\b', r'\bdone\b', r'\bpass[ed]?\b',
        r'\bverified\b', r'\bcorrect\b', r'\btrue\b', r'\b100%\b',
        r'\bfinished\b', r'\bexit code 0\b', r'\bzero exit\b',
        r'\btest passed\b', r'\ball tests\b', r'\bverification\b',
        r'\bcreated\b', r'\bgenerated\b', r'\bexported\b', r'\bsaved\b',
        r'\bchecksum', r'\bhash', r'\bmd5', r'\bsha', r'\bdecrypted\b',
        r'\bencrypted\b', r'\bkey', r'\bcertificate\b', r'\bvalid\b'
    ]
    
    # Error indicators (decrease Q-value)
    error_patterns = [
        r'\berror\b', r'\bfail[ed]?\b', r'\bexception\b', r'\bdenied\b',
        r'\brefused\b', r'\binvalid\b', r'\bwrong\b', r'\bincorrect\b',
        r'\bpermission denied\b', r'\bno such file\b', r'\bnot found\b',
        r'\bexit code [1-9]\b', r'\bnon-zero exit\b', r'\btraceback\b',
        r'\bfatal\b', r'\bcritical\b', r'\babort\b', r'\btimeout\b',
        r'\bconnection refused\b', r'\btimeout\b', r'\bfailed\b'
    ]
    
    # Progress indicators
    progress_patterns = [
        r'\bstep [0-9]+\b', r'\b[0-9]+ of [0-9]+\b', r'\b([0-9]+)%\b',
        r'\bremaining\b', r'\bprocessing\b', r'\bloading\b', r'\breading\b'
    ]
    
    # Count successes
    success_count = 0
    for pattern in success_patterns:
        if re.search(pattern, next_lower):
            success_count += 1
    
    # Count errors
    error_count = 0
    for pattern in error_patterns:
        if re.search(pattern, next_lower):
            error_count += 1
    
    # Count progress indicators
    progress_count = 0
    for pattern in progress_patterns:
        if re.search(pattern, next_lower):
            progress_count += 1
    
    # Apply adjustments
    if error_count > 0:
        q_value -= 0.3 * min(error_count, 3)
    
    if success_count > 0:
        q_value += 0.2 * min(success_count, 5)
    
    if progress_count > 0:
        q_value += 0.05 * progress_count
    
    # Check for state change (action made progress)
    if next_state != state:
        q_value += 0.05
    
    # Check for action relevance (commands that typically make progress)
    progress_actions = ['cd', 'mkdir', 'touch', 'cp', 'mv', 'rm', 'cat', 'echo',
                        'grep', 'find', 'ls', 'chmod', 'chown', 'tar', 'zip',
                        'pip', 'conda', 'apt', 'yum', 'docker', 'git', 'ssh',
                        'scp', 'rsync', 'python', 'bash', 'sh', 'run', 'execute',
                        'solve', 'complete', 'verify', 'test', 'check']
    
    action_has_progress = any(a in action_lower for a in progress_actions)
    if action_has_progress:
        q_value += 0.05
    
    # Check for completion-like patterns in next_state
    completion_keywords = ['final', 'result', 'output', 'answer', 'solution',
                           'summary', 'report', 'status', 'completed']
    completion_count = sum(1 for kw in completion_keywords if kw in next_lower)
    if completion_count > 0:
        q_value += 0.1 * min(completion_count, 3)
    
    # Check if next_state appears to be a goal state (contains answer-like content)
    goal_indicators = [r'\banswer:\b', r'\bresult:\b', r'\bsolution:\b',
                       r'\boutput:\b', r'\bfinal:\b', r'\b✓\b', r'\b✔\b']
    for pattern in goal_indicators:
        if re.search(pattern, next_lower):
            q_value += 0.3
    
    # Clamp Q-value to reasonable range
    q_value = max(0.0, min(1.5, q_value))
    
    return float(q_value)