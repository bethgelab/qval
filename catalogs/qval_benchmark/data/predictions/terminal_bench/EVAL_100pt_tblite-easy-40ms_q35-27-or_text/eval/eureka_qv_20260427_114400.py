def signal_function(state: str, action: str, next_state: str):
    import re
    
    # Goal completion indicators
    goal_keywords = ['success', 'done', 'completed', 'result', 'output', 'finished', 'pass', 'verified', 'ok', 'true', 'match', 'found', 'correct', 'valid', 'accepted', 'test passed', 'verification', 'solution', '✓', '✔', '✅', '100%', 'all tests', 'passed', 'solved', 'resolved', 'fixed', 'correctly', 'working', 'task complete', 'final', 'end', 'finish', 'success!', '✓✓✓', 'all done', 'test pass', 'passed all', 'verification passed', 'all tests passed']
    
    # Test pass indicators (stronger signal)
    test_pass_keywords = ['test passed', 'all tests passed', 'verification passed', 'verified successfully', '✓✓✓', '100%', 'passed all', 'test pass', 'passed verification', 'all pass', 'success!', '✅', 'all done', 'solved successfully', 'tests passed', 'passed tests', 'all tests pass', 'verification success']
    
    # Failure indicators
    failure_keywords = ['failed', 'failure', 'error', 'exception', 'crash', 'segfault', 'timeout', 'timeout error', 'assertion failed', 'assert failed', 'test failed', 'tests failed', 'verification failed', 'not passed', 'did not pass', 'incorrect', 'wrong', 'invalid', 'mismatch', 'mismatched', 'unexpected', 'uncaught', 'unhandled', 'runtime error', 'fatal', 'fatal error', 'panic', 'abort', 'killed', 'oom', 'out of memory', 'permission denied', 'not found', 'cannot', 'refused', 'broken', 'corrupted']
    
    # Partial completion indicators
    partial_keywords = ['partial', 'some', 'only', 'partially', 'incomplete', 'failed to', 'could not', 'unable to', 'not all', 'missing', 'incomplete', 'partial success', 'some tests', 'partial result']
    
    goal_count = sum(1 for kw in goal_keywords if kw.lower() in state.lower())
    test_pass_count = sum(1 for kw in test_pass_keywords if kw.lower() in state.lower())
    failure_count = sum(1 for kw in failure_keywords if kw.lower() in state.lower())
    partial_count = sum(1 for kw in partial_keywords if kw.lower() in state.lower())
    
    # Estimate completion percentage based on state features
    # More nuanced calibration for incomplete tasks
    base_completion = min(goal_count * 0.12, 0.6)
    if test_pass_count > 0:
        base_completion = max(base_completion, 0.85)
    
    # Step budget awareness (40 steps max)
    # Estimate progress based on state characteristics
    state_length = len(state)
    # Rough estimate: longer output usually means more work done
    estimated_progress = min(max(state_length / 1000.0, 0.05), 0.95)
    
    # Goal score with better calibration
    goal_score = min(goal_count * 0.22, 0.65)
    
    # Test pass bonus
    test_pass_bonus = 0.0
    if test_pass_count > 0:
        test_pass_bonus = 0.28
    elif goal_count > 3:
        test_pass_bonus = 0.14
    elif goal_count > 1:
        test_pass_bonus = 0.08
    
    # Failure score - calibrated to not be too harsh
    failure_score = 0.0
    if failure_count > 0:
        failure_score = -min(failure_count, 5) * 0.18
    
    # Error severity scoring
    critical_errors = ['segfault', 'crash', 'timeout', 'exception', 'traceback', 'fatal', 'abort', 'killed', 'oom', 'out of memory', 'error: ', 'error:', 'fail:', 'failed:', 'runtime error', 'system error', 'critical', 'panic', 'fatal error', 'assertion failed', 'assert failed', 'uncaught exception', 'unhandled exception']
    moderate_errors = ['permission denied', 'not found', 'cannot', 'invalid', 'denied', 'refused', 'broken', 'corrupted', 'mismatch', 'wrong', 'incorrect', 'test failed', 'tests failed']
    minor_errors = ['warning', 'warn', 'deprecated', 'note', 'info', 'debug']
    
    critical_count = sum(1 for kw in critical_errors if kw.lower() in state.lower())
    moderate_count = sum(1 for kw in moderate_errors if kw.lower() in state.lower())
    minor_count = sum(1 for kw in minor_errors if kw.lower() in state.lower())
    
    error_score = 0.0
    error_score -= min(critical_count, 3) * 0.22
    error_score -= min(moderate_count, 2) * 0.09
    error_score -= min(minor_count, 3) * 0.025
    
    # Partial completion penalty
    partial_penalty = 0.0
    if partial_count > 0 and goal_count > 0:
        partial_penalty = -0.12 * min(partial_count, 2)
    
    # Action quality assessment
    action_score = 0.0
    action_lower = action.lower().strip()
    
    if len(action) < 2:
        action_score -= 0.08
    elif len(action) > 300:
        action_score -= 0.03
    
    productive_commands = ['cat', 'ls', 'grep', 'find', 'mkdir', 'touch', 'echo', 'cd', 'python', 'pip', 'install', 'run', 'execute', 'chmod', 'chown', 'cp', 'mv', 'rm', 'wget', 'curl', 'ssh', 'git', 'make', 'build', 'test', 'verify', 'submit', 'check', 'validate', 'diff', 'sed', 'awk', 'sort', 'head', 'tail', 'wc', 'file', 'tar', 'zip', 'unzip', 'gunzip', 'gzip', 'openssl', 'ssh-keygen', 'docker', 'apt', 'yum', 'conda', 'virtualenv', 'pipenv', 'jupyter', 'scipy', 'numpy', 'pandas', 'sklearn', 'tensorflow', 'pytorch', 'train', 'evaluate', 'predict', 'save', 'load', 'read', 'write', 'parse', 'decode', 'encode', 'decrypt', 'encrypt', 'hash', 'sha', 'md5', 'base64', 'solve', 'compute', 'calculate', 'generate', 'create', 'print', 'return', 'export', 'import']
    
    for cmd in productive_commands:
        if cmd in action_lower:
            action_score += 0.12
            break
    
    unproductive_patterns = ['!!!', '...', '???', 'help', 'clear', 'reset', 'exit', 'quit', 'logout', 'bye', 'goodbye', 'nothing', 'wait', 'sleep']
    for pattern in unproductive_patterns:
        if pattern in action_lower:
            action_score -= 0.08
            break
    
    # Progress detection between state and next_state
    progress_score = 0.0
    if next_state != state:
        state_lower = state.lower()
        next_lower = next_state.lower()
        
        for kw in goal_keywords:
            if kw in next_lower and kw not in state_lower:
                progress_score += 0.28
                break
        
        for kw in critical_errors:
            if kw in next_lower and kw not in state_lower:
                progress_score -= 0.28
                break
        
        # Length-based progress
        if len(next_state) > len(state):
            progress_score += 0.05
    
    # Terminal state bonus
    terminal_keywords = ['submit', 'finish', 'complete', 'done', 'end', 'final', 'verify', 'check', 'validate', 'test', 'run', 'execute', 'solve', 'confirm', 'accept']
    if any(kw in action_lower for kw in terminal_keywords):
        if test_pass_count > 0:
            terminal_bonus = 0.65
        elif goal_count > 2:
            terminal_bonus = 0.45
        elif goal_count > 0:
            terminal_bonus = 0.22
        else:
            terminal_bonus = 0.10
    else:
        terminal_bonus = 0.0
    
    # Step efficiency bonus based on estimated progress
    step_bonus = estimated_progress * 0.12
    
    # Combine all components
    total = goal_score + failure_score + error_score + action_score + progress_score + terminal_bonus + step_bonus + test_pass_bonus + partial_penalty
    
    # Calibrate for incomplete tasks with mixed signals
    if failure_count > 0 and goal_count > 0:
        # Mixed state: scale down but don't penalize too heavily
        total = total * 0.88
    elif failure_count > 0 and goal_count == 0:
        # Only failures: more aggressive penalty
        total = total * 0.65
    elif test_pass_count > 0:
        # Clear success: slight boost
        total = min(total * 1.05, 1.0)
    
    # Clamp to [0, 1] range
    total = max(0.0, min(1.0, total))
    
    return total, {
        "goal_score": goal_score,
        "failure_score": failure_score,
        "error_score": error_score,
        "action_score": action_score,
        "progress_score": progress_score,
        "terminal_bonus": terminal_bonus,
        "step_bonus": step_bonus,
        "test_pass_bonus": test_pass_bonus,
        "partial_penalty": partial_penalty,
    }