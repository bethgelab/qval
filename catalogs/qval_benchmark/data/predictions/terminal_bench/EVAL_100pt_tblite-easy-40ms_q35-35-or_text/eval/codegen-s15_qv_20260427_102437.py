import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Normalize strings for case-insensitive pattern matching
    next_lower = next_state.lower()
    action_lower = action.lower()
    
    # Define success and failure patterns to detect terminal states
    success_patterns = [
        r'\bpass\b', r'\bsuccess\b', r'\bverified\b',
        r'\bcompleted\b', r'\btask done\b', r'\bok\b'
    ]
    failure_patterns = [
        r'\berror\b', r'\bfailed\b', r'\bexception\b',
        r'\btraceback\b', r'\bfatal\b', r'\bpermission denied\b'
    ]
    
    # Check for explicit success indicators in next_state
    for pattern in success_patterns:
        if re.search(pattern, next_lower):
            return 1.0
    
    # Check for explicit failure indicators in next_state
    for pattern in failure_patterns:
        if re.search(pattern, next_lower):
            return 0.0
            
    # Check for shell prompt (indicates command finished executing cleanly)
    # Matches common prompts: $, #, >, or user@host:~
    prompt_pattern = r'(\$|#|>|@\w+:~)\s*$'
    if re.search(prompt_pattern, next_state):
        # Base score for a clean prompt suggests progress
        score = 0.3
        
        # Boost score if action looks like a finalization step
        final_keywords = ['submit', 'verify', 'test', 'run', 'make', 'install', 'deploy']
        if any(kw in action_lower for kw in final_keywords):
            score = 0.5
        
        # Penalize if the state indicates waiting for interaction (e.g., password)
        if 'password' in next_lower or 'continue' in next_lower:
            score = 0.1
            
        return score
    else:
        # Command might still be running or state is unstable/intermediate
        return 0.1