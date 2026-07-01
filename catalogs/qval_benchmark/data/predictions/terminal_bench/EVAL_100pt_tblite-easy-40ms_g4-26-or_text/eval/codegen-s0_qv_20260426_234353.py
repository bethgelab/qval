import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The Q-value represents the estimated expected return (probability of success).
    """
    ns = next_state.lower()
    act = action.lower()

    # 1. Terminal Success Indicators
    # If the output contains strings typical of a successful verification or command completion.
    success_patterns = [
        r'success',
        r'passed',
        r'\bok\b',
        r'done',
        r'completed',
        r'0 errors',
        r'no errors',
        r'all tests passed',
        r'test passed'
    ]
    if any(re.search(p, ns) for p in success_patterns):
        return 1.0

    # 2. Terminal Failure Indicators
    # If the output contains strings typical of command or script failures.
    failure_patterns = [
        r'failed',
        r'not found',
        r'no such file',
        r'permission denied',
        r'command not command',
        r'syntax error',
        r'exception',
        r'traceback',
        r'usage:',
        r'invalid argument',
        r'\berror\b'
    ]

    is_failure = False
    for p in failure_patterns:
        if re.search(p, ns):
            # Heuristic: avoid false positives for search/grep commands where 'error' is a match.
            if 'error' in p and any(cmd in act for cmd in ['grep', 'find', 'search']):
                continue
            is_failure = True
            break
    
    if is_failure:
        return 0.0

    # 3. Intermediate Progress/Heuristic Estimates
    # If no explicit success/failure is detected, we estimate based on the nature of the action.
    
    # Solving Actions: Actions that likely move the environment toward the goal (modifications/installations).
    # Using word boundaries to avoid false positives.
    solving_regex = r'\b(pip|apt|mkdir|cp|mv|chmod|python|gcc|git|wget|curl|touch|sed|awk|install)\b'
    
    # Information/Investigative Actions: Actions that gather info about the current state.
    info_regex = r'^(ls|cat|grep|pwd|find|which|dir|type|echo|head|tail)\b'
    
    # Navigation Actions: Actions that change context (directories).
    nav_regex = r'^(cd|pwd)\b'

    if re.search(solving_regex, act):
        # Command likely attempts to resolve a task requirement or fix a dependency.
        return 0.7
    elif re.search(info_regex, act):
        # Command is likely checking the current state/environment.
        return 0.5
    elif re.search(nav_regex, act):
        # Command is likely navigating the file system.
        return 0.6
    else:
        # Default estimate for unknown or neutral actions.
        return 0.5