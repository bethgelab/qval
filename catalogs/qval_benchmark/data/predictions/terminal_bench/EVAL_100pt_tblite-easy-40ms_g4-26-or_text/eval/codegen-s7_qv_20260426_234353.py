import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The estimate is based on identifying success, failure, and the type of command executed.
    """
    # 1. Normalization
    next_s = next_state.strip().lower()
    action_s = action.strip().lower()
    state_s = state.strip().lower()
    
    # 2. Immediate Success/Failure Detection
    # Success indicators: High confidence the task is complete
    success_terms = ["success", "passed", "0 errors", "no errors", "all tests passed", "verified", "completed"]
    if any(term in next_s for term in success_terms):
        return 1.0
        
    # Failure indicators: High confidence the action failed or caused an error
    # We look for standard shell/language error messages
    failure_terms = [
        "command not found", 
        "no such file", 
        "permission denied", 
        "syntax error", 
        "not a directory", 
        "invalid syntax", 
        "error:", 
        "exception:",
        "not recognized"
    ]
    for term in failure_terms:
        if term in next_s:
            # Check if the error term is inside quotes (e.g., grep "error"), 
            # which might indicate it's just output rather than a shell error.
            if not re.search(r"['\"]" + re.escape(term) + r"['\"]", next_s):
                return 0.0

    # 3. Heuristic Categorization of Actions
    # Priority: Verification > Setup > Information Gathering > Default
    
    # Commands likely used to check solution or run scripts
    is_verify = any(kw in action_s for kw in ["test", "verify", "check", "run", "python", "bash", "./"])
    
    # Commands used to modify or prepare the environment
    is_setup = any(kw in action_s for kw in [
        "mkdir", "pip", "apt", "chmod", "chown", "cd", "cp", "mv", "rm", 
        "install", "wget", "curl", "sudo", "apt-get"
    ])
    
    # Commands used to explore the file system or environment
    is_info = any(kw in action_s for kw in [
        "ls", "cat", "grep", "find", "pwd", "dir", "which", "echo", "type"
    ])

    # 4. Output Analysis
    # Measure how much new information was provided by the command
    # Comparing stripped versions helps ignore differences in trailing whitespace/prompts
    output_diff = len(next_state.strip()) - len(state.strip())
    has_meaningful_output = output_diff > 2

    # 5. Q-Value Assignment
    # We return expected cumulative reward. 
    # If it's a verification step and didn't error, it's highly likely to be a success.
    if is_verify:
        return 0.85 if has_meaningful_output else 0.5
    
    # If it's a setup command, it's a productive step toward the goal.
    if is_setup:
        return 0.45 if has_meaningful_output else 0.35
    
    # If it's an information command, it's a way to gain knowledge to progress.
    if is_info:
        return 0.4 if has_meaningful_output else 0.2
    
    # Default for other commands (e.g., moving within directories, etc.)
    if has_meaningful_output:
        return 0.3
    
    return 0.1