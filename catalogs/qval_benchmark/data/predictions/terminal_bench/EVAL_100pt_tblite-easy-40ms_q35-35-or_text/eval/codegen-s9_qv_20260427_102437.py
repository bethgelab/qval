import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Define keywords indicative of task progress or failure
    success_keywords = [
        'success', 'done', 'complete', 'verified', 'passed', 'created', 
        'installed', 'solution', 'flag', 'ok', 'result', 'verifier passed',
        'all tests passed', 'task completed', 'flag found'
    ]
    error_keywords = [
        'error', 'fail', 'exception', 'traceback', 'denied', 'not found', 
        'missing', 'invalid', 'undefined', 'crash', 'segfault', 'timeout',
        'permission denied', 'syntax error', 'import error', 'no such file'
    ]
    final_action_keywords = ['submit', 'verify', 'check', 'test', 'exit', 'quit', 'done']
    
    # Normalize for case-insensitive matching
    next_lower = next_state.lower()
    action_lower = action.lower()
    
    # Count pattern matches
    success_count = sum(1 for kw in success_keywords if kw in next_lower)
    error_count = sum(1 for kw in error_keywords if kw in next_lower)
    is_final_action = any(kw in action_lower for kw in final_action_keywords)
    
    # Base Q-value estimate
    score = 0.5
    
    # Penalize empty or unresponsive next states
    if not next_state.strip():
        score -= 0.2
    else:
        # Reward success indicators
        score += success_count * 0.15
        
        # Penalize error indicators
        score -= error_count * 0.25
        
        # Adjust for final actions (e.g., submit/verify)
        if is_final_action:
            # Errors on final actions are critical
            if error_count > 0:
                score -= 0.3
            # Success on final actions is highly valuable
            if success_count > 0:
                score += 0.2

    # Clamp score to valid probability range [0.0, 1.0]
    return float(max(0.0, min(1.0, score)))