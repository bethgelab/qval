import re
import math
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for TerminalBench environment based on state analysis.
    Returns float between 0.0 and 1.0 representing expected return.
    """
    # Base score starts at 0.5 (neutral expectation)
    score = 0.5
    
    # Check for success/completion indicators (high reward)
    success_patterns = [
        r'verifier.*pass', r'PASSED', r'PASSED', r'success', r'complete',
        r'All tests passed', r'OK', r'correct', r'solved', r'correct answer',
        r'Congratulations', r'Congratulations!', r'You have completed'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            score = 0.95
            break
    
    # Check for error/failure indicators (low reward)
    error_patterns = [
        r'error', r'fail', r'failed', r'exception', r'not found',
        r'Permission denied', r'access denied', r'invalid', r'incorrect',
        r'wrong', r'mismatch', r'not correct', r'failed to',
        r'No such file', r'command not found', r'permission'
    ]
    
    error_count = 0
    for pattern in error_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            error_count += 1
    
    if error_count >= 3:
        score = 0.1
    elif error_count >= 1:
        score = max(score, 0.3)
    
    # Check for progress indicators (moderate reward)
    progress_patterns = [
        r'task', r'problem', r'question', r'challenge', r'step',
        r'command', r'output', r'output:', r'output=', r'running',
        r'executing', r'creating', r'writing', r'saving', r'generated'
    ]
    
    progress_count = sum(1 for p in progress_patterns if re.search(p, state, re.IGNORECASE))
    if progress_count >= 3:
        score = max(score, 0.6)
    elif progress_count >= 1:
        score = max(score, 0.4)
    
    # Check for task completion markers
    completion_markers = [
        r'flag', r'CTF', r'solution', r'answer', r'result',
        r'output file', r'output.txt', r'results'
    ]
    
    for marker in completion_markers:
        if re.search(marker, state, re.IGNORECASE):
            score = max(score, 0.7)
            break
    
    # Action quality analysis
    action_lower = action.lower()
    
    # Constructive actions (positive)
    constructive_actions = [
        'echo', 'cat', 'ls', 'pwd', 'cd', 'mkdir', 'touch',
        'python', 'python3', 'node', 'npm', 'pip', 'git',
        'grep', 'find', 'curl', 'wget', 'ssh', 'scp',
        'chmod', 'chown', 'cp', 'mv', 'rm', 'tar', 'gzip',
        'openssl', 'base64', 'md5sum', 'sha256sum', 'cat'
    ]
    
    for act in constructive_actions:
        if act in action_lower:
            score = max(score, 0.55)
            break
    
    # Destructive/restart actions (neutral to slightly negative)
    destructive_actions = ['rm -rf', 'rm -r', 'kill', 'pkill', 'sudo', 'su']
    for act in destructive_actions:
        if act in action_lower:
            score = min(score, 0.7)
            break
    
    # Next state improvement (if next_state shows progress)
    if next_state and next_state != state:
        # Check if next state has more progress indicators
        next_progress = sum(1 for p in progress_patterns if re.search(p, next_state, re.IGNORECASE))
        state_progress = sum(1 for p in progress_patterns if re.search(p, state, re.IGNORECASE))
        
        if next_progress > state_progress:
            score = min(score + 0.1, 0.85)
        elif next_progress < state_progress:
            score = max(score - 0.1, 0.2)
    
    # Penalty for very long state (might indicate getting stuck)
    if len(state) > 5000:
        score = min(score, 0.6)
    
    # Ensure score is in valid range
    score = max(0.0, min(1.0, score))
    
    return score