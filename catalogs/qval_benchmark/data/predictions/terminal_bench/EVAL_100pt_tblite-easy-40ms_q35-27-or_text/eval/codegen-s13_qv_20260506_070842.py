def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    base_value = 0.5
    
    # Analyze next_state for error indicators
    error_patterns = ['error', 'failed', 'exception', 'traceback', 'fatal', 'denied', 'permission', 'cannot', 'invalid', 'refused']
    for pattern in error_patterns:
        if pattern in next_state.lower():
            base_value -= 0.25
    
    # Analyze next_state for success indicators
    success_patterns = ['success', 'done', 'completed', 'pass', 'ok', 'finished', 'ready', 'accepted', 'verified']
    for pattern in success_patterns:
        if pattern in next_state.lower():
            base_value += 0.2
    
    # Check if action produced meaningful state change
    if state != next_state:
        base_value += 0.15
    
    # Penalize excessively verbose output (may indicate issues)
    if len(next_state) > 5000:
        base_value -= 0.1
    
    # Check for explicit completion signals
    completion_signals = ['✓', '✔', 'PASS', 'OK', 'SUCCESS', 'true', '1', 'completed:', 'finished:']
    for signal in completion_signals:
        if signal in next_state:
            base_value += 0.2
    
    # Check for step progress indicators
    step_match = re.search(r'step\s+(\d+)\s+of\s+(\d+)', next_state, re.IGNORECASE)
    if step_match:
        current_step = int(step_match.group(1))
        total_steps = int(step_match.group(2))
        if total_steps > 0:
            progress = current_step / total_steps
            base_value += progress * 0.3
    
    # Check for X/Y progress pattern
    progress_match = re.search(r'(\d+)/(\d+)', next_state)
    if progress_match and not step_match:
        current = int(progress_match.group(1))
        total = int(progress_match.group(2))
        if total > 0 and current <= total:
            progress = current / total
            base_value += progress * 0.2
    
    # Check if action appears to be a submission/verification attempt
    action_lower = action.lower()
    submit_patterns = ['submit', 'verify', 'check', 'test', 'run', 'execute', 'finish', 'end', 'complete']
    for pattern in submit_patterns:
        if pattern in action_lower:
            base_value += 0.15
            break
    
    # Check for file completion indicators
    file_patterns = ['created', 'generated', 'written', 'saved', 'exported']
    for pattern in file_patterns:
        if pattern in next_state.lower():
            base_value += 0.1
    
    # Check for process/task completion
    process_patterns = ['exited', 'terminated', 'completed', 'finished', 'done']
    for pattern in process_patterns:
        if pattern in next_state.lower():
            base_value += 0.15
    
    # Clamp value to reasonable range
    return max(0.0, min(2.0, base_value))