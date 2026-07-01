def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Keywords indicating task completion (High Q-value)
    success_patterns = [
        "success", "completed", "done", "saved", "sent", 
        "created", "added", "confirmed", "finished", "posted"
    ]
    
    # Keywords indicating errors or dead ends (Low Q-value)
    error_patterns = [
        "error", "fail", "404", "not found", "unable", 
        "blocked", "failed", "exception"
    ]
    
    # Keywords indicating intermediate progress (Medium Q-value)
    progress_patterns = [
        "form", "input", "dialog", "modal", "list", "item", 
        "calendar", "todo", "message", "event"
    ]
    
    # Base score initialization
    score = 0.0
    
    # 1. Check for immediate goal achievement in next_state
    is_success = False
    for pattern in success_patterns:
        if pattern in next_state.lower():
            is_success = True
            break
    
    if is_success:
        # Goal achieved, reward is 1.0, discounting applied implicitly by high value
        return 0.95
    
    # 2. Check for errors in current state
    is_error = False
    for pattern in error_patterns:
        if pattern in state.lower():
            is_error = True
            break
    
    if is_error:
        # Stuck in error state, low probability of success
        return 0.05
    
    # 3. Check for no progress (state unchanged)
    if state == next_state:
        # Action did not change the state, likely inefficient
        return 0.05
    
    # 4. Evaluate action type
    action_lower = action.lower()
    if "noop" in action_lower:
        score = 0.1
    elif "scroll" in action_lower:
        score = 0.2
    elif "click" in action_lower or "fill" in action_lower:
        score = 0.4
    else:
        score = 0.3
    
    # 5. Adjust based on state content (progress indicators)
    next_state_lower = next_state.lower()
    for pattern in progress_patterns:
        if pattern in next_state_lower:
            score += 0.1
    
    # 6. Check for loading states (waiting for result)
    if "loading" in next_state_lower or "processing" in next_state_lower:
        score = max(0.0, min(1.0, score + 0.1))
    
    # 7. Clamp score between 0.0 and 1.0
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
        
    return score