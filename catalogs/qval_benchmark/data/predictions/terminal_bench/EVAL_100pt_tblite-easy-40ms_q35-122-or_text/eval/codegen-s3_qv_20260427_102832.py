def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Base Q-value estimate
    q_value = 0.0
    
    # Normalize text for analysis
    state_text = state.lower()
    action_text = action.lower()
    next_text = next_state.lower()
    
    # Success/completion indicators in next state
    success_patterns = [
        'success', 'complete', 'done', 'finished', 'pass', 'verified',
        'correct', 'task complete', 'all tests passed', 'verification passed',
        'exit code 0', '0 errors', 'no errors', 'completed successfully'
    ]
    
    # Failure/error indicators
    failure_patterns = [
        'error', 'failed', 'fail', 'denied', 'permission', 'wrong',
        'incorrect', 'not found', 'no such file', 'syntax error',
        'command not found', 'exit code 1', 'exit code 2', 'exception'
    ]
    
    # Progress indicators
    progress_patterns = [
        'created', 'modified', 'updated', 'installed', 'copied', 'moved',
        'added', 'removed', 'written', 'saved', 'generated', 'compiled'
    ]
    
    # Count success indicators in next_state
    success_count = sum(1 for p in success_patterns if p in next_text)
    q_value += success_count * 0.25
    
    # Penalize failure indicators
    failure_count = sum(1 for p in failure_patterns if p in next_text)
    q_value -= failure_count * 0.15
    
    # Reward progress
    progress_count = sum(1 for p in progress_patterns if p in next_text)
    q_value += progress_count * 0.1
    
    # Check if action is productive (file operations, navigation, inspection)
    productive_commands = [
        'mkdir', 'touch', 'cp', 'mv', 'cat', 'echo', 'cd', 'ls', 'find',
        'grep', 'chmod', 'chown', 'ln', 'tar', 'zip', 'unzip', 'nano',
        'vim', 'vimdiff', 'sed', 'awk', 'python', 'pip', 'apt', 'apt-get',
        'install', 'build', 'compile', 'make', 'gcc', 'g++'
    ]
    if any(cmd in action_text for cmd in productive_commands):
        q_value += 0.1
    
    # Penalize destructive or risky actions without confirmation
    risky_commands = ['rm -rf', 'rm -r', 'dd ', 'format', 'wipe', 'mkfs']
    if any(risk in action_text for risk in risky_commands):
        q_value -= 0.1
    
    # Check for state change (action had effect)
    if next_text.strip() and next_text != state_text:
        # Action produced output or changed state
        q_value += 0.05
    
    # Penalize empty or unchanged states
    if not next_text.strip() or next_text == state_text:
        q_value -= 0.05
    
    # Bonus for task-specific indicators (common in terminal benchmarks)
    task_indicators = [
        'flag', 'solution', 'answer', 'result', 'output', 'answer is',
        'the flag is', 'submit', 'verified', 'checksum', 'hash'
    ]
    task_count = sum(1 for ind in task_indicators if ind in next_text)
    q_value += task_count * 0.15
    
    # Check if we're getting closer (next state has more content than before)
    if len(next_text) > len(state_text) and len(next_text) < 10000:
        q_value += 0.05
    
    # Penalize very long error messages
    if failure_count > 2 or 'traceback' in next_text:
        q_value -= 0.1
    
    # Clamp to reasonable Q-value range
    q_value = max(-1.0, min(1.5, q_value))
    
    return float(q_value)