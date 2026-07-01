import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on heuristic analysis of state, action, and next_state.
    Returns a float in range [-1.0, 1.0] representing estimated expected return.
    """
    base_value = 0.0
    
    # Check for goal completion indicators in next_state
    success_patterns = [
        r'success', r'complete', r'completed', r'pass', r'verified',
        r'ok', r'ok\.', r'0 errors', r'exit code 0', r'exit status 0',
        r'✓', r'[✓]', r'PASS', r'Test passed', r'All tests passed',
        r'Task completed', r'Done', r'Finished'
    ]
    
    # Check for failure indicators in next_state
    failure_patterns = [
        r'error', r'fail', r'failed', r'error:', r'Exception', r'Error',
        r'Exit code 1', r'exit status 1', r'No such file', r'Permission denied',
        r'command not found', r'not found', r'failed', r'FAIL', r'✗', r'[✗]',
        r'FAILED', r'ERROR', r'RuntimeError', r'ValueError', r'KeyError'
    ]
    
    # Check for progress indicators in next_state
    progress_patterns = [
        r'writing', r'created', r'made', r'generated', r'exported',
        r'saved', r'uploaded', r'installed', r'configured', r'initialized',
        r'running', r'active', r'enabled', r'connected', r'linked'
    ]
    
    # Check for task-specific indicators in state
    task_keywords = [
        'task', 'objective', 'goal', 'verify', 'test', 'check',
        'solution', 'answer', 'output', 'result', 'output', 'print'
    ]
    
    # Count pattern matches (case-insensitive)
    def count_pattern_matches(text, patterns):
        text_lower = text.lower()
        matches = sum(1 for p in patterns if re.search(p, text_lower, re.IGNORECASE))
        return matches
    
    success_count = count_pattern_matches(next_state, success_patterns)
    failure_count = count_pattern_matches(next_state, failure_patterns)
    progress_count = count_pattern_matches(next_state, progress_patterns)
    task_count = count_pattern_matches(state, task_keywords)
    
    # Weighted scoring
    if success_count > 0:
        base_value += 0.8 * success_count
    if failure_count > 0:
        base_value -= 0.6 * failure_count
    if progress_count > 0:
        base_value += 0.3 * progress_count
    if task_count > 0:
        base_value += 0.1
    
    # Normalize to reasonable range
    if base_value > 1.0:
        base_value = 1.0
    if base_value < -1.0:
        base_value = -1.0
    
    # Consider action length as rough proxy for complexity
    action_length = len(action)
    if action_length > 50:
        base_value -= 0.1  # Long commands might indicate complex operations
    
    return float(base_value)