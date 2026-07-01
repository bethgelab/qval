def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for TerminalBench terminal tasks.
    
    Uses heuristic analysis of state representations to estimate
    expected return without simulation or lookahead.
    """
    
    # Normalize strings for analysis
    next_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Success indicators - presence suggests we're on track
    success_keywords = [
        'success', 'passed', 'verified', 'complete', 'done',
        'test passed', 'all tests', 'verification', 'correct',
        'answer', 'solution', 'flag', 'challenge', 'accepted',
        'output', 'result', 'found', 'created', 'written'
    ]
    
    # Failure indicators - presence suggests problems
    failure_keywords = [
        'error', 'failed', 'fail', 'incorrect', 'wrong',
        'exception', 'traceback', 'denied', 'not found',
        'permission', 'denied', 'invalid', 'syntax', 'undefined'
    ]
    
    # Count indicators in next_state
    success_count = sum(1 for kw in success_keywords if kw in next_lower)
    failure_count = sum(1 for kw in failure_keywords if kw in next_lower)
    
    # Base value from success/failure signals
    if success_count >= 2:
        base_value = 0.9
    elif success_count == 1:
        base_value = 0.7
    elif failure_count >= 2:
        base_value = 0.1
    elif failure_count == 1:
        base_value = 0.3
    else:
        base_value = 0.5
    
    # Check if action was a productive terminal command
    productive_commands = [
        'ls', 'cat', 'echo', 'cd', 'mkdir', 'rm', 'cp', 'mv',
        'python', 'bash', 'sh', 'grep', 'find', 'chmod', 'chown',
        'tar', 'zip', 'unzip', 'wc', 'head', 'tail', 'sort',
        'awk', 'sed', 'curl', 'wget', 'git', 'ssh', 'scp',
        'nano', 'vim', 'touch', 'pwd', 'date', 'whoami', 'id',
        'ps', 'kill', 'top', 'df', 'du', 'free', 'uname',
        'apt', 'pip', 'npm', 'make', 'cmake', 'gcc', 'g++'
    ]
    
    action_parts = action_lower.split()
    is_productive = any(cmd in action_parts for cmd in productive_commands)
    
    if is_productive:
        base_value += 0.1
    
    # Check for progress (new content generated)
    state_len = len(state)
    next_len = len(next_state)
    
    if next_len > 0 and next_len > state_len:
        # New output suggests productive action
        progress_ratio = min(1.0, (next_len - state_len) / max(1, state_len))
        base_value += 0.1 * progress_ratio
    
    # Penalize empty or suspicious actions
    if len(action_lower.strip()) < 3:
        base_value -= 0.15
    
    if action_lower in ['help', 'q', 'quit', 'exit', 'clear']:
        base_value -= 0.1
    
    # Penalize if state seems to regress (similar to initial state)
    if state_len > 100 and next_len < state_len * 0.5:
        base_value -= 0.1
    
    # Clamp to valid Q-value range [0, 1]
    q_value = max(0.0, min(1.0, base_value))
    
    return q_value