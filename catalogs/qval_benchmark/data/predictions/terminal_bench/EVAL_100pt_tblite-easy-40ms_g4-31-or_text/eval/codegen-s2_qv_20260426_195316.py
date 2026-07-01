def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a reinforcement learning environment
    based on shell state transitions.
    """
    s_low = state.lower()
    ns_low = next_state.lower()
    a_low = action.lower()
    
    # Keywords indicative of success or failure
    pos_keywords = ["success", "verified", "correct", "completed", "passed", "done", "found"]
    neg_keywords = ["error", "fail", "denied", "not found", "invalid", "syntax error", "exception", "cannot find"]
    
    # Analyze transitions in state indicators
    had_pos = any(p in s_low for p in pos_keywords)
    has_pos = any(p in ns_low for p in pos_keywords)
    had_neg = any(n in s_low for n in neg_keywords)
    has_neg = any(n in ns_low for n in neg_keywords)
    
    moved_to_pos = has_pos and not had_pos
    moved_to_neg = has_neg and not had_neg
    fixed_neg = not has_neg and had_neg
    
    # Special handling for submission actions
    # Submission is the critical terminal step; its outcome heavily determines the return.
    if "submit" in a_low:
        if has_pos:
            return 1.0
        if has_neg:
            return 0.0
        # If the submit action was taken but the outcome is ambiguous, 
        # provide a moderate value.
        return 0.5
    
    # General intermediate step estimation
    # Baseline Q-value represents the expected discounted return for an average action.
    q = 0.3
    
    # Significant progress: Moving from a state without success markers to one with them.
    if moved_to_pos:
        q += 0.4
    # Significant regression: Introducing new errors into the state.
    elif moved_to_neg:
        q -= 0.2
    # Incremental progress: Removing existing errors.
    elif fixed_neg:
        q += 0.2
    
    # Efficiency adjustment: Commands that are purely exploratory (e.g., ls, pwd) 
    # are valued slightly lower than those that modify the system.
    exploration_cmds = ["ls", "pwd", "whoami", "history", "clear", "date"]
    action_stripped = a_low.strip()
    if any(action_stripped == cmd or action_stripped.startswith(cmd + " ") for cmd in exploration_cmds):
        q -= 0.1
        
    # Stagnation penalty: If the state doesn't change meaningfully, it's a lower-value move.
    if ns_low == s_low:
        q -= 0.1

    # Ensure the Q-value is bounded between 0.0 and 1.0
    return max(0.0, min(1.0, q))