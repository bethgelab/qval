import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a 
    terminal-based environment. The Q-value represents the expected 
    discounted cumulative reward, which is binary (1.0 for success).
    """
    # Baseline Q-value
    score = 0.5
    
    # Keywords indicating progress or success
    success_patterns = [
        r'\bpassed\b', 
        r'\bsuccess\b', 
        r'\bcorrect\b', 
        r'\bverified\b', 
        r'\bcompleted\b', 
        r'\bdone\b', 
        r'all tests passed', 
        r'exit code 0'
    ]
    
    # Keywords indicating failure or errors
    failure_patterns = [
        r'\berror\b', 
        r'\bfailed\b', 
        r'\bnot found\b', 
        r'\bdenied\b', 
        r'\binvalid\b', 
        r'\bsyntax error\b', 
        r'\bexception\b', 
        r'no such file', 
        r'traceback'
    ]
    
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # 1. Terminal/Verification Action Analysis
    # Actions like 'submit' or 'verify' are high-stakes; their outcomes are strong predictors.
    is_verification = any(kw in action_lower for kw in ['submit', 'verify', 'test', 'check'])
    
    if is_verification:
        has_success = any(re.search(p, next_state_lower) for p in success_patterns)
        has_failure = any(re.search(p, next_state_lower) for p in failure_patterns)
        if has_success:
            return 0.95
        if has_failure:
            return 0.05
    
    # 2. General Outcome Analysis
    # Look for success/failure indicators in the resulting state.
    if any(re.search(p, next_state_lower) for p in success_patterns):
        score += 0.2
    if any(re.search(p, next_state_lower) for p in failure_patterns):
        score -= 0.2
        
    # 3. Action Intent Analysis
    # We distinguish between exploration, construction, and potential errors.
    
    # Exploration commands provide information but don't directly solve the problem.
    info_cmds = ['ls', 'pwd', 'whoami', 'id', 'uname', 'cat']
    # Constructive commands move the state toward a goal.
    constructive_cmds = ['vim', 'nano', 'sed', 'awk', 'gcc', 'python', 'pip', 'mkdir', 'touch', 'cp', 'mv', 'grep', 'find', 'chmod', 'chown']
    
    # Extract the primary command from the action string
    cmd_parts = action_lower.split()
    if cmd_parts:
        cmd_start = cmd_parts[0]
        if cmd_start in info_cmds:
            score -= 0.05  # Slight penalty for simple exploration
        elif cmd_start in constructive_cmds:
            score += 0.1   # Bonus for using tools that change the environment
            
    # 4. Redundancy and Stagnation
    # If the state hasn't changed significantly, the action was likely ineffective.
    if next_state == state:
        score -= 0.2
    elif len(next_state) - len(state) < 5:
        # Small change might be a silent command (like mkdir), but frequently indicates no progress.
        # We only penalize this if it wasn't a constructive command.
        if cmd_parts and cmd_parts[0] not in constructive_cmds:
            score -= 0.1

    # Ensure the result is clamped within [0.0, 1.0]
    return max(0.0, min(1.0, score))