def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value
    q_value = 0.5
    
    # Normalize text for analysis
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Goal/completion indicators (strong positive signal)
    goal_keywords = ['success', 'complete', 'done', 'finished', 'passed', 
                     'verified', 'correct', 'match', 'found', 'ok', 'true',
                     'solution', 'answer', 'result', 'output', 'generated',
                     'created', 'written', 'saved', 'exported', 'compiled',
                     'built', 'installed', 'configured', 'enabled', 'active']
    
    # Error/failure indicators (negative signal)
    error_keywords = ['error', 'fail', 'failed', 'exception', 'traceback',
                      'wrong', 'incorrect', 'invalid', 'not found', 'permission',
                      'denied', 'refused', 'abort', 'interrupted', 'crash',
                      'segfault', 'timeout', 'refused', 'unable', 'cannot']
    
    # Progress indicators (moderate positive)
    progress_keywords = ['running', 'processing', 'loading', 'computing',
                         'analyzing', 'searching', 'building', 'installing',
                         'downloading', 'extracting', 'unpacking', 'reading',
                         'writing', 'saving', 'creating', 'generating']
    
    # Count occurrences
    goal_count = sum(1 for kw in goal_keywords if kw in next_state_lower)
    error_count = sum(1 for kw in error_keywords if kw in next_state_lower)
    progress_count = sum(1 for kw in progress_keywords if kw in next_state_lower)
    
    # Apply weighted adjustments
    q_value += goal_count * 0.15
    q_value -= error_count * 0.2
    q_value += progress_count * 0.05
    
    # Check if action produced meaningful output (state changed)
    if len(next_state) > len(state) * 1.05:
        q_value += 0.05
    
    # Check for productive command patterns
    productive_patterns = ['python', 'pip', 'git', 'curl', 'wget', 'tar', 
                           'mkdir', 'touch', 'cp', 'mv', 'rm', 'cat', 'grep',
                           'find', 'awk', 'sed', 'chmod', 'chown', 'ssh',
                           'scp', 'rsync', 'docker', 'docker-compose', 'make',
                           'gcc', 'g++', 'npm', 'node', 'java', 'mvn', 'gradle']
    
    for pattern in productive_patterns:
        if pattern in action_lower:
            q_value += 0.03
            break
    
    # Penalize potentially wasteful actions
    wasteful_patterns = ['sleep', 'wait', 'pause', 'cat /dev/null', 'true', 'false']
    for pattern in wasteful_patterns:
        if pattern in action_lower:
            q_value -= 0.1
            break
    
    # Check for file creation indicators (common in terminal tasks)
    file_patterns = ['.py', '.txt', '.json', '.csv', '.log', '.sh', '.md',
                     '.yaml', '.yml', '.xml', '.sql', '.env', '.conf', '.cfg']
    for pattern in file_patterns:
        if pattern in next_state_lower and pattern not in state_lower:
            q_value += 0.08
            break
    
    # Check for command execution success patterns
    if next_state_lower.count('$') > state_lower.count('$'):
        q_value += 0.02  # New prompt suggests command completed
    
    # Penalize if next state contains fewer goal indicators than current
    state_goal_count = sum(1 for kw in goal_keywords if kw in state_lower)
    if state_goal_count > 0 and goal_count < state_goal_count:
        q_value -= 0.1
    
    # Clamp to valid Q-value range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value