def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for TerminalBench terminal task environment.
    
    Q(s,a) represents expected discounted cumulative reward.
    Binary reward: 1.0 on success, 0.0 otherwise.
    Episode limit: 40 steps.
    """
    
    # Normalize to lowercase for pattern matching
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()
    
    # Success/progress indicators - these suggest we're getting closer to goal
    success_terms = [
        'success', 'done', 'completed', 'pass', 'passed', 'verified',
        'correct', 'output', 'result', 'answer', 'solution', 'test',
        'created', 'written', 'saved', 'found', 'located', 'resolved'
    ]
    
    # Error/failure indicators - these suggest we're off track
    error_terms = [
        'error', 'fail', 'failed', 'failure', 'not found', 'denied',
        'invalid', 'syntax', 'missing', 'unable', 'permission',
        'command not found', 'no such', 'does not exist'
    ]
    
    # Calculate success evidence from next_state
    success_evidence = sum(1 for term in success_terms if term in ns)
    
    # Calculate error evidence from next_state
    error_evidence = sum(1 for term in error_terms if term in ns)
    
    # Calculate progress evidence from state (current state shows progress)
    progress_evidence = sum(1 for term in success_terms if term in s)
    
    # Action quality scoring - some actions are more likely to progress
    action_score = 0.5  # baseline
    
    # Command-based actions are generally productive
    productive_cmds = ['cat ', 'grep ', 'find ', 'ls ', 'cd ', 'mkdir ', 
                       'cp ', 'mv ', 'rm ', 'chmod ', 'python', 'bash',
                       'write', 'create', 'solve', 'implement', 'fix']
    for cmd in productive_cmds:
        if cmd in a:
            action_score = max(action_score, 0.7)
            break
    
    # Actions with file paths or specific targets are more purposeful
    if '/' in a or '.' in a.split()[-1] if a.split() else False:
        action_score = min(1.0, action_score + 0.1)
    
    # Actions that seem random or exploratory score lower
    exploratory = ['whoami', 'pwd', 'history', 'clear', 'exit']
    for exp in exploratory:
        if exp in a:
            action_score = max(0.0, action_score - 0.1)
    
    # Base Q-value from success/error balance
    base_q = 0.4
    
    if success_evidence > 0:
        base_q += 0.15 * success_evidence
        base_q = min(1.0, base_q)
    
    if error_evidence > 0:
        base_q -= 0.1 * error_evidence
        base_q = max(0.0, base_q)
    
    # Progress in current state indicates we're on the right track
    if progress_evidence > 0:
        base_q += 0.05 * progress_evidence
    
    # Combine with action quality
    q_value = 0.6 * base_q + 0.4 * action_score
    
    # Check for explicit completion signals
    if any(term in ns for term in ['all tests passed', 'verification passed', 'task complete']):
        q_value = 0.95
    
    # Ensure valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)