def signal_function(state: str, action: str, next_state: str) -> float:
    # Base score for neutral progress
    score = 0.5
    
    # Normalize strings for case-insensitive matching
    s_lower = state.lower()
    a_lower = action.lower()
    ns_lower = next_state.lower()
    
    # Success indicators (task completion signals)
    success_indicators = [
        "verification passed", "task completed", "congratulations", 
        "score: 100", "success", "done", "exit code 0", "your answer is"
    ]
    
    # Failure indicators (error signals)
    failure_indicators = [
        "error", "failed", "exception", "traceback", "permission denied", 
        "no such file", "not found", "connection refused", "timeout"
    ]
    
    # Dangerous action indicators
    dangerous_actions = ["rm -rf /", "sudo rm -rf", "mkfs", "dd if=/dev/zero"]
    
    # Check for success first (highest priority)
    for indicator in success_indicators:
        if indicator in ns_lower:
            return 0.95
            
    # Check for dangerous actions (safety priority)
    for act in dangerous_actions:
        if act in a_lower:
            return 0.0
            
    # Check for failure indicators
    for indicator in failure_indicators:
        if indicator in ns_lower:
            score = 0.1
            break
    
    # Check for stagnation (no progress)
    if len(next_state) == 0 and len(state) > 0:
        score = min(score, 0.2)
    elif next_state == state:
        score = min(score, 0.3)
        
    # Clamp and return score as float
    return float(max(0.0, min(1.0, score)))