import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a terminal environment.
    The Q-value represents the estimated probability of eventually reaching the goal.
    """
    # Initial baseline value for any action taken.
    score = 0.1
    
    # 1. Detect shell-level errors in the resulting state.
    # Common patterns indicating a failed command or configuration error.
    error_patterns = [
        r"command not found",
        r"Permission denied",
        r"No such file or directory",
        r"SyntaxError",
        r"Invalid argument",
        r"failed with exit code",
        r"error:",
        r"cannot open",
        r"not found",
        r"usage:", # Often indicates wrong arguments provided to a command
    ]
    
    is_error = False
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            is_error = True
            score -= 0.3
            break
            
    # 2. Detect success markers in the resulting state.
    # These are strong indicators that the action led to a goal or a milestone.
    success_patterns = [
        r"successfully",
        r"completed",
        r"done",
        r"passed",
        r"correctly",
        r"all tests passed",
        r"verification successful",
        r"goal reached",
    ]
    
    is_success = False
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            is_success = True
            score += 0.5
            break

    # 3. Reward "productive" actions that did not result in an error.
    # Common tools used in system admin, ML, crypto, and data tasks.
    productive_tools = [
        'grep', 'find', 'cat', 'ls', 'vim', 'nano', 'python', 'perl', 'ruby', 
        'gcc', 'make', 'pip', 'apt', 'chmod', 'chown', 'ssh', 'curl', 'wget',
        'openssl', 'gpg', 'base64', 'systemctl', 'journalctl', 'df', 'du', 
        'top', 'ps', 'netstat', 'iptables', 'torch', 'tensorflow', 'numpy', 'pandas'
    ]
    
    # Check if the action involves a productive tool and didn't fail.
    if not is_error:
        if any(tool in action.lower() for tool in productive_tools):
            score += 0.1
            
    # 4. Analyze "Finality" - actions meant to submit or verify the final result.
    if "submit" in action.lower() or "verify" in action.lower() or "test" in action.lower():
        if is_success:
            score += 0.4  # Highly likely to be the final optimal step
        elif is_error:
            score -= 0.2  # Failed attempt at submission
        else:
            score += 0.1   # Attempting verification is better than doing nothing

    # 5. Penalize stagnation.
    # If the next state is virtually identical to the previous state (excluding the prompt),
    # the action was likely redundant or a no-op.
    # We use a simple heuristic: if the length difference is very small and no new tokens appeared.
    if abs(len(next_state) - len(state)) < 10 and not is_success:
        score -= 0.1

    # Clamp the final Q-value between 0.0 and 1.0.
    return max(0.0, min(1.0, score))