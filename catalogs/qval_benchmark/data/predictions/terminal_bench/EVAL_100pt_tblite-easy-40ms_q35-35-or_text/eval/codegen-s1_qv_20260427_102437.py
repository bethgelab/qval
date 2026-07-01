def signal_function(state: str, action: str, next_state: str) -> float:
    # Success indicators for TerminalBench tasks
    success_keywords = [
        'success', 'pass', 'verified', 'flag', 'solution', 'completed',
        'congratulations', 'accepted', 'correct', 'score', 'output', 'done',
        'flag{', 'test passed', 'score: 100'
    ]
    
    # Error indicators for TerminalBench tasks
    error_keywords = [
        'error', 'failed', 'denied', 'exception', 'traceback', 'not found',
        'syntax error', 'permission denied', 'undefined', 'crash', 'invalid',
        'command not found', 'permission denied', 'no such file'
    ]
    
    # Base Q-value estimate (neutral probability of success)
    q_value = 0.5
    
    # Normalize strings for case-insensitive matching
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Count success indicators in next_state
    success_matches = sum(1 for p in success_keywords if p in next_state_lower)
    if success_matches > 0:
        # High reward for reaching goal state
        q_value += min(0.4, success_matches * 0.1)
    
    # Count error indicators in next_state
    error_matches = sum(1 for p in error_keywords if p in next_state_lower)
    if error_matches > 0:
        # Low reward for error states
        q_value -= min(0.4, error_matches * 0.1)
    
    # Progress heuristic: longer next_state often indicates more context gathered
    # (TerminalBench tasks often accumulate output)
    if len(next_state) > len(state):
        q_value += 0.05
    elif len(next_state) < len(state):
        # Significant loss of state might indicate clearing or error
        q_value -= 0.02
    
    # Action heuristics
    # Favorable actions that typically gather info or execute logic
    constructive_actions = ['cat', 'ls', 'grep', 'python', 'bash', 'run', 'solve', 'submit', 'check']
    for act in constructive_actions:
        if act in action_lower:
            q_value += 0.02
            break
            
    # Unfavorable actions without clear success context
    destructive_actions = ['rm -rf', 'rm -r', 'reboot', 'shutdown']
    for act in destructive_actions:
        if act in action_lower:
            if success_matches == 0:
                q_value -= 0.05
            break
    
    # Clamp to valid probability range [0.0, 1.0]
    return max(0.0, min(1.0, q_value))