def signal_function(state: str, action: str, next_state: str) -> float:
    # Success indicators for task completion
    success_keywords = ["added", "created", "saved", "sent", "done", "success", 
                        "completed", "posted", "confirmed", "updated", "removed"]
    
    # Failure indicators
    error_keywords = ["error", "failed", "invalid", "unauthorized", "not found", 
                      "unable", "cannot", "404", "500"]
    
    # Progress indicators for intermediate steps
    progress_keywords = ["submit", "save", "send", "next", "confirm", "dialog", 
                         "modal", "form", "input", "button", "link", "task", "event"]
    
    # 1. Check for immediate goal achievement
    next_lower = next_state.lower()
    for kw in success_keywords:
        if kw in next_lower:
            return 1.0
            
    # 2. Check for errors
    for kw in error_keywords:
        if kw in next_lower:
            return 0.0
            
    # 3. Analyze Action Quality
    action_lower = action.lower()
    action_score = 0.0
    if "noop" in action_lower:
        action_score = 0.0
    elif "scroll" in action_lower:
        action_score = 0.1
    elif "press" in action_lower:
        action_score = 0.2
    elif "click" in action_lower:
        action_score = 0.3
    elif "fill" in action_lower:
        action_score = 0.4
    else:
        action_score = 0.1
        
    # 4. Analyze State Progress
    state_lower = state.lower()
    state_progress = sum(1 for kw in progress_keywords if kw in state_lower)
    next_progress = sum(1 for kw in progress_keywords if kw in next_lower)
    
    state_score = 0.0
    if next_progress > state_progress:
        state_score = 0.3
    elif next_progress == state_progress:
        state_score = 0.1
    else:
        state_score = 0.0
        
    # 5. Stagnation Penalty
    if action_score < 0.2 and next_progress <= state_progress:
        state_score -= 0.1
        
    # 6. Combine and Clamp
    q_value = action_score + state_score
    return max(0.0, min(1.0, q_value))