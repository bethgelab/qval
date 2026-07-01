def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Normalize to lowercase for pattern matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_lower = next_state.lower()
    
    # Completion indicators - these suggest task success
    success_patterns = [
        'success', 'completed', 'passed', 'verified', 'test passed',
        'all tests passed', 'verification passed', 'done', 'finished'
    ]
    
    # Failure indicators - these suggest errors or problems
    error_patterns = [
        'error', 'failed', 'failure', 'exception', 'traceback',
        'not found', 'cannot', 'permission denied', 'no such file',
        'syntax error', 'invalid', 'unresolved'
    ]
    
    # Productive action patterns - actions that typically advance progress
    productive_actions = [
        'create', 'write', 'edit', 'compile', 'run', 'execute',
        'test', 'verify', 'check', 'submit', 'save', 'copy',
        'move', 'rename', 'chmod', 'chown', 'install', 'pip',
        'python', 'gcc', 'make', 'build'
    ]
    
    # Count matches for each category
    success_count = sum(1 for p in success_patterns if p in next_lower)
    error_count = sum(1 for p in error_patterns if p in next_lower)
    productive_count = sum(1 for p in productive_actions if p in action_lower)
    
    # Check for specific completion markers
    is_completed = any(p in next_lower for p in success_patterns)
    is_failed = any(p in next_lower for p in error_patterns)
    
    # Terminal state handling
    if is_completed:
        return 1.0
    if is_failed:
        return 0.0
    
    # Base Q-value starts at neutral
    q_value = 0.5
    
    # Adjust based on error presence (errors reduce value)
    if error_count > 0:
        q_value -= 0.15 * error_count
    
    # Adjust based on success indicators (even partial progress is good)
    if success_count > 0:
        q_value += 0.2 * success_count
    
    # Adjust based on action quality (productive actions are better)
    if productive_count > 0:
        q_value += 0.1 * productive_count
    
    # Check for progress indicators in state transitions
    # More output lines in next_state often means more work done
    state_lines = len(state.split('\n'))
    next_lines = len(next_state.split('\n'))
    if next_lines > state_lines + 5:
        q_value += 0.05  # Significant output suggests progress
    
    # Check for file operations that indicate progress
    file_indicators = ['file', 'created', 'saved', 'written', 'modified']
    if any(ind in next_lower for ind in file_indicators):
        q_value += 0.05
    
    # Penalize if action seems destructive without clear purpose
    destructive_patterns = ['rm ', 'delete', 'remove', 'wipe']
    if any(p in action_lower for p in destructive_patterns) and is_failed:
        q_value -= 0.2
    
    # Ensure value stays in reasonable range [0, 1.2]
    q_value = max(0.0, min(1.2, q_value))
    
    return float(q_value)