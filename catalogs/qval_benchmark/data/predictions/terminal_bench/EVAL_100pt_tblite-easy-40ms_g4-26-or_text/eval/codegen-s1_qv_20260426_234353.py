import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a, s') for a terminal task.
    The estimate is based on identifying success, failure, or progress in the transition.
    """
    # 1. Direct Goal Detection
    # If the output contains patterns of a found secret or flag.
    if re.search(r"(flag\{|key\{|secret=)", next_state, re.IGNORECASE):
        return 0.95
        
    # 2. Action Categorization
    # Determine if the action is likely a verification or test-running command.
    # We use broad keywords to identify commands like 'python test.py', './verify.sh', etc.
    test_patterns = r"(test|verify|check|run|python|bash|sh|pytest|unittest|npm test|make test|./verify)"
    is_test_action = bool(re.search(test_patterns, action, re.IGNORECASE))
    
    # 3. Output Analysis
    # Detect error/failure messages.
    error_patterns = r"(error|fail(ed)?|not found|no such|denied|invalid|command not found|permission denied)"
    is_error = bool(re.search(error_patterns, next_state, re.IGNORECASE))
    
    # Detect success/completion messages using word boundaries to avoid false positives.
    success_patterns = r"\b(pass(ed)?|success|ok|done|complete|correct|verified)\b"
    is_success = bool(re.search(success_patterns, next_state, re.IGNORECASE))
    
    # 4. Q-Value Estimation Logic
    
    # Case A: The action was intended to verify the solution.
    if is_test_action:
        if is_error:
            return 0.0
        if is_success:
            return 1.0
        return 0.5  # Uncertain result
        
    # Case B: Not a test action, but the output shows immediate success/error.
    if is_error:
        return 0.05
    
    if is_success:
        return 0.8
        
    # Case C: Not a test action, check if the command produced useful information.
    # If the output is non-empty and not an error, it's likely an exploratory step.
    if next_state.strip():
        return 0.4
        
    # Default low value for empty/uninformative transitions.
    return 0.1