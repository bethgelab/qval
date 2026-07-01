import re
import json

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next_state in a terminal management environment.
    
    Logic:
    1. Check if the next_state contains explicit success indicators (e.g., "Success", "PASS", "Verified", "Goal reached").
    2. Check if the next_state contains explicit failure indicators (e.g., "Error", "Failed", "Traceback", "Permission denied").
    3. If the action looks like a "submit" or "finish" command, the Q-value is heavily weighted by the immediate success/failure of the next_state.
    4. If the action is an intermediate command (cd, ls, cat, etc.), the Q-value is based on progress indicators (e.g., file sizes increasing, error messages decreasing, or specific progress logs).
    5. Default to a neutral value if no clear signal is found.
    """
    
    # Success keywords
    success_keywords = ["success", "pass", "verified", "goal reached", "all tests passed", "exit code 0", "true", "ok"]
    # Failure keywords
    fail_keywords = ["error", "failed", "traceback", "permission denied", "no such file", "command not found", "exit code 1", "false", "fail", "exception"]
    # Intermediate/Progress keywords
    progress_keywords = ["created", "updated", "wrote", "copied", "moved", "saved", "processing", "complete", "done", "finished", "exported", "encoded"]
    # Submit/Finish indicators in action
    submit_actions = ["submit", "finish", "submit_solution", "exit", "q", "clear"]
    
    # Helper to check if a keyword exists (case-insensitive)
    def has_keyword(text, keywords):
        text_lower = text.lower()
        for k in keywords:
            if k in text_lower:
                return True
        return False
    
    # Helper to count keyword occurrences (simple heuristic for strength)
    def count_keywords(text, keywords):
        text_lower = text.lower()
        count = 0
        for k in keywords:
            count += text_lower.count(k)
        return count
    
    # Determine if this is a terminal action
    is_terminal_action = any(action.lower().startswith(s) for s in submit_actions)
    
    # Calculate immediate reward signal based on next_state
    immediate_success = has_keyword(next_state, success_keywords)
    immediate_failure = has_keyword(next_state, fail_keywords)
    has_progress = has_keyword(next_state, progress_keywords)
    
    q_value = 0.0
    
    if immediate_success:
        q_value = 0.95 # High confidence of success
    elif immediate_failure:
        q_value = -0.95 # High confidence of failure
    elif is_terminal_action:
        # If it's a submit action, check for partial success or neutral
        if has_progress(next_state, progress_keywords):
            q_value = 0.3 # Likely submitted but not yet verified, or partial success
        elif not immediate_failure:
            q_value = 0.1 # Neutral, assuming it ran without error
        else:
            q_value = -0.5 # Submitted and likely failed
    elif has_progress:
        # Intermediate step showing progress
        q_value = 0.2
    elif immediate_failure:
        q_value = -0.8
    else:
        # Neutral state, assume potential for success but no guarantee
        q_value = 0.0
        
    # Boost Q-value if the state itself looks very clean (e.g., contains specific task completion markers)
    if has_keyword(state, success_keywords):
        q_value = min(1.0, q_value + 0.1)
        
    # Cap the value
    return max(-1.0, min(1.0, q_value))