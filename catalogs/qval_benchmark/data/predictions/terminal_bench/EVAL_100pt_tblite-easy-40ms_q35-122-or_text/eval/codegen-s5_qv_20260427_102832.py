def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Convert to lowercase for easier pattern matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_lower = next_state.lower()
    
    # Base Q-value starts at 0.5 (neutral)
    q_value = 0.5
    
    # Success indicators in next_state
    success_patterns = [
        'success', 'completed', 'done', 'passed', 'ok', 'ok!',
        'test passed', 'all tests passed', 'verification', 'verified',
        'solution', 'answer', 'output', 'result', 'found',
        'flag', 'ctf', 'correct', 'valid', 'accepted'
    ]
    
    error_patterns = [
        'error', 'failed', 'exception', 'syntax error', 'undefined',
        'not found', 'permission denied', 'denied', 'invalid',
        'traceback', 'segmentation fault', 'core dumped', 'timeout'
    ]
    
    progress_patterns = [
        'created', 'written', 'saved', 'compiled', 'built', 'installed',
        'downloaded', 'extracted', 'copied', 'moved', 'modified',
        'updated', 'installed', 'configured', 'generated', 'decoded',
        'encrypted', 'decrypted', 'parsed', 'processed', 'transformed'
    ]
    
    # Check for success indicators in next_state
    success_count = sum(1 for p in success_patterns if p in next_lower)
    if success_count > 0:
        q_value += 0.15 * success_count
        q_value = min(q_value, 1.0)
    
    # Check for error indicators in next_state
    error_count = sum(1 for p in error_patterns if p in next_lower)
    if error_count > 0:
        q_value -= 0.2 * error_count
        q_value = max(q_value, 0.0)
    
    # Check for progress indicators
    progress_count = sum(1 for p in progress_patterns if p in next_lower)
    if progress_count > 0:
        q_value += 0.1 * progress_count
    
    # Estimate progress based on state length difference
    state_len = len(state)
    next_len = len(next_state)
    if next_len > state_len:
        # State grew, likely some progress
        growth_ratio = min((next_len - state_len) / max(state_len, 1), 0.5)
        q_value += growth_ratio * 0.1
    
    # Evaluate action quality
    constructive_actions = [
        'write', 'create', 'make', 'build', 'compile', 'run',
        'execute', 'test', 'check', 'verify', 'solve', 'find',
        'decode', 'decrypt', 'extract', 'parse', 'process',
        'submit', 'answer', 'print', 'output'
    ]
    
    destructive_actions = [
        'rm ', 'delete', 'remove', 'kill', 'abort', 'cancel',
        'clear', 'erase', 'destroy'
    ]
    
    if any(a in action_lower for a in constructive_actions):
        q_value += 0.05
    if any(a in action_lower for a in destructive_actions):
        q_value -= 0.1
    
    # Check if action seems to be a command (starts with common shell commands)
    command_patterns = ['cd ', 'ls ', 'cat ', 'echo ', 'python', 'bash', 'sh ', './']
    if any(action_lower.startswith(p) for p in command_patterns):
        q_value += 0.02
    
    # Penalize empty or very short actions
    if len(action.strip()) < 3:
        q_value -= 0.1
    
    # Penalize if state shows we're stuck (repeated similar content)
    if state_len > 0 and next_len > 0:
        if abs(next_len - state_len) < 10 and error_count == 0:
            # Minimal change without errors might indicate stuck
            q_value -= 0.05
    
    # Bonus for actions that look like they're completing the task
    completion_keywords = ['submit', 'finish', 'done', 'complete', 'final']
    if any(k in action_lower for k in completion_keywords):
        q_value += 0.1
    
    # Ensure Q-value is in valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return float(q_value)