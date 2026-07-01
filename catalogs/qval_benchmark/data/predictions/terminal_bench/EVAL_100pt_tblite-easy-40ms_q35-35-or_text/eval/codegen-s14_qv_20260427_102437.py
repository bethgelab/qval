def signal_function(state: str, action: str, next_state: str) -> float:
    # Base score representing a neutral starting point
    score = 0.2

    # Normalize strings for case-insensitive matching
    next_lower = next_state.lower()
    action_lower = action.lower()

    # Check for success indicators in the next state
    success_keywords = ['success', 'pass', 'verified', 'flag', 'correct', 
                        'solution', 'answer', 'done', 'complete', 'accepted', 'ok', 'valid']
    for keyword in success_keywords:
        if keyword in next_lower:
            score += 0.85
            break

    # Check for error indicators in the next state
    error_keywords = ['error', 'failed', 'denied', 'not found', 'permission', 
                      'syntax error', 'exception', 'fail', 'traceback', 'bug']
    for keyword in error_keywords:
        if keyword in next_lower:
            score -= 0.6
            break

    # Analyze action for intent
    if 'submit' in action_lower or 'verify' in action_lower or 'check' in action_lower:
        score += 0.15
    elif 'rm -rf' in action_lower or 'reboot' in action_lower or 'shutdown' in action_lower:
        score -= 0.3
    elif any(cmd in action_lower for cmd in ['ls', 'cat', 'pwd', 'grep', 'echo', 'head', 'tail', 'find', 'which']):
        score += 0.05

    # Clamp the score between 0.0 and 1.0
    return max(0.0, min(1.0, score))