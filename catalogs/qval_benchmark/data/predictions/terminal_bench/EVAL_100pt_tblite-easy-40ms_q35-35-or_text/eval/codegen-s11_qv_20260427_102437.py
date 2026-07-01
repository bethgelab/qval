import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Combine text for analysis, focusing on next_state for immediate feedback
    context = (state + " " + action + " " + next_state).lower()
    
    # Success indicators (Task completion)
    success_patterns = [
        r'passed', r'verified', r'success', r'ok', r'complete', r'done',
        r'all tests passed', r'task passed', r'solution verified', r'exit code 0'
    ]
    is_success = any(re.search(p, context) for p in success_patterns)
    
    # Failure indicators (Task failure)
    failure_patterns = [
        r'task failed', r'verification failed', r'tests failed', r'exit code 1',
        r'permission denied', r'error', r'failed', r'exception', r'traceback'
    ]
    is_failure = any(re.search(p, context) for p in failure_patterns)
    
    # Progress indicators (Moving towards goal)
    progress_patterns = [
        r'created', r'saved', r'wrote', r'installed', r'compiled',
        r'generated', r'submitted', r'solution', r'output', r'file'
    ]
    progress_count = sum(1 for p in progress_patterns if re.search(p, context))
    
    # Action validity check (Simple syntax balance)
    action_valid = True
    if action.count('"') % 2 != 0:
        action_valid = False
    if action.count("'") % 2 != 0:
        action_valid = False
        
    # Calculate Q-value
    if is_success:
        return 1.0
    elif is_failure:
        return 0.0
    else:
        # Intermediate state estimation
        base_score = 0.5
        
        # Bonus for progress indicators found
        progress_bonus = min(progress_count * 0.1, 0.3)
        
        # Penalty for action syntax issues
        syntax_penalty = 0.2 if not action_valid else 0.0
        
        score = base_score + progress_bonus - syntax_penalty
        
        # Clamp between 0.0 and 1.0
        return max(0.0, min(1.0, score))