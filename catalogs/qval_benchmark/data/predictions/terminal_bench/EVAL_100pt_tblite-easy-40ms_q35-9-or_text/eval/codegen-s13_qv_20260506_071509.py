import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal shell environment.
    
    Logic:
    1. If the next_state indicates success (contains 'success', 'passed', 'True', '0 errors', 'Goal reached', etc.), 
       assign a high positive value.
    2. If the next_state indicates failure (contains 'error', 'failed', 'False', 'Traceback', 'Permission denied', etc.), 
       assign a low negative value.
    3. If the next_state indicates progress or a neutral state (contains 'done', 'complete', 'export', 'cd', 'ls', etc.), 
       assign a moderate positive value reflecting progress towards the goal.
    4. If the state or next_state is empty or malformed, return a neutral or slightly negative value.
    
    This heuristic avoids recursion and simulation, relying solely on keyword analysis of the text strings.
    """
    
    # Keywords indicating success
    success_keywords = ['success', 'passed', 'true', '0 errors', 'goal reached', 'verification passed', 'test passed', 'ok']
    # Keywords indicating failure
    failure_keywords = ['error', 'failed', 'false', 'traceback', 'permission denied', 'command not found', 'syntax error', 'invalid', 'fail']
    # Keywords indicating neutral/progress (commands executed without immediate success/failure feedback)
    progress_keywords = ['done', 'complete', 'export', 'cd', 'ls', 'mkdir', 'touch', 'cat', 'echo', 'python', 'bash', 'sh', 'grep', 'find', 'sed', 'awk', 'pip', 'apt', 'git']
    
    next_lower = next_state.lower()
    state_lower = state.lower()
    
    # Check for success indicators in next_state
    if any(kw in next_lower for kw in success_keywords):
        return 0.95
    
    # Check for failure indicators in next_state
    if any(kw in next_lower for kw in failure_keywords):
        return -0.8
    
    # Check for progress indicators
    if any(kw in next_lower for kw in progress_keywords):
        return 0.2
    
    # Check if state or next_state is empty
    if not next_state or not state:
        return -0.1
    
    # Check for specific success patterns like exit code 0 in a summary line
    if re.search(r'exit\s+code\s+[:\s]+0', next_lower):
        return 0.9
    
    # Check for specific failure patterns like non-zero exit code
    if re.search(r'exit\s+code\s+[:\s]+[1-9]', next_lower):
        return -0.7
    
    # Default fallback for unknown states
    return 0.0