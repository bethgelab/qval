def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for a terminal shell task based on state analysis.
    Returns a float between 0.0 and 1.0 representing expected return quality.
    """
    import re
    
    # Handle empty or invalid inputs
    if not state or not action or not next_state:
        return 0.2
    
    state_lower = state.lower()
    action_lower = action.lower()
    next_lower = next_state.lower()
    
    # Base value for valid action
    q_value = 0.3
    
    # Penalize error indicators in next_state
    error_patterns = ['error', 'failed', 'fail', 'permission denied', 
                      'no such file', 'not found', 'invalid', 'traceback']
    error_count = sum(1 for p in error_patterns if p in next_lower)
    if error_count > 0:
        q_value = max(0.0, q_value - 0.3 * error_count)
    
    # Reward success indicators
    success_patterns = ['success', 'completed', 'done', 'passed', 'test passed',
                        'verification', '✓', '✔', 'correct']
    success_count = sum(1 for p in success_patterns if p in next_lower)
    if success_count > 0:
        q_value = min(1.0, q_value + 0.4 * success_count)
    
    # Reward productive shell commands
    productive_commands = ['cd', 'ls', 'cat', 'echo', 'mkdir', 'touch', 'cp', 
                           'mv', 'rm', 'chmod', 'grep', 'find', 'python', 'bash',
                           'make', 'gcc', 'git', 'pip', 'install', 'run']
    has_productive = any(cmd in action_lower for cmd in productive_commands)
    if has_productive:
        q_value = min(1.0, q_value + 0.15)
    
    # Reward file/directory operations (indicates progress)
    file_patterns = ['.py', '.txt', '.json', '.md', '.sh', '.csv', '.log',
                     'file', 'directory', 'created', 'modified', 'saved']
    file_progress = sum(1 for p in file_patterns if p in next_lower)
    if file_progress > 0:
        q_value = min(1.0, q_value + 0.1 * file_progress)
    
    # Reward state change (indicates action had effect)
    state_change = abs(len(next_state) - len(state))
    if state_change > 50:
        q_value = min(1.0, q_value + 0.1)
    
    # Penalize no change (action may have been ineffective)
    if state_change < 10 and error_count == 0:
        q_value = max(0.1, q_value - 0.1)
    
    # Reward task-specific keywords (common in TerminalBench)
    task_keywords = ['solution', 'answer', 'output', 'result', 'submit',
                     'task', 'challenge', 'benchmark', 'test case']
    task_progress = sum(1 for k in task_keywords if k in next_lower)
    if task_progress > 0:
        q_value = min(1.0, q_value + 0.15 * task_progress)
    
    # Bonus for commands that look like final submission
    submission_patterns = ['submit', 'final', 'complete', 'answer', 'output']
    is_submission = any(p in action_lower for p in submission_patterns)
    if is_submission and success_count > 0:
        q_value = 1.0
    
    return max(0.0, min(1.0, q_value))