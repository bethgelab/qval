def signal_function(state: str, action: str, next_state: str):
    progress_score = 0.0
    safety_score = 0.0
    efficiency_score = 0.0
    statefulness_score = 0.0
    completion_bonus = 0.0
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Analyze bid count changes as proxy for interaction progress
    state_bid_count = state_lower.count("bid")
    next_bid_count = next_state_lower.count("bid")
    
    bid_change = next_bid_count - state_bid_count
    
    # Positive bid change indicates new elements appeared (likely progress)
    if bid_change > 0:
        progress_score += min(bid_change * 0.06, 0.25)
    elif bid_change < 0:
        progress_score -= min(abs(bid_change) * 0.02, 0.1)
    
    # Detect task completion signals more comprehensively
    completion_keywords = ["saved", "done", "complete", "success", "added", "created", 
                          "task completed", "goal reached", "form submitted", "event added",
                          "confirmation", "successfully", "finished", "confirm", "submitted"]
    has_completion_next = any(kw in next_state_lower for kw in completion_keywords)
    has_completion_state = any(kw in state_lower for kw in completion_keywords)
    
    # Completion bonus: spike when we're about to complete
    if has_completion_next and not has_completion_state:
        completion_bonus = 0.75
        progress_score += 0.45
    elif has_completion_next:
        completion_bonus = 0.35
        progress_score += 0.2
    elif has_completion_state:
        # Already completed - action shouldn't add much value
        completion_bonus = -0.15
        progress_score -= 0.08
    
    # Action type analysis with better granularity
    if "noop" in action_lower:
        safety_score -= 0.25
        efficiency_score -= 0.18
    elif "scroll" in action_lower:
        safety_score -= 0.05
        efficiency_score += 0.02
    elif "fill" in action_lower:
        progress_score += 0.35
        efficiency_score += 0.22
    elif "click" in action_lower:
        # Check if clicking on likely goal elements
        goal_click_keywords = ["submit", "save", "add", "create", "send", "post", "confirm", "done", "finish", "ok"]
        if any(kw in action_lower for kw in goal_click_keywords):
            progress_score += 0.42
            completion_bonus += 0.25
        else:
            progress_score += 0.18
            efficiency_score += 0.08
    elif "press" in action_lower:
        progress_score += 0.12
        efficiency_score += 0.04
    
    # Detect error/failure signals in next state
    error_indicators = ["error", "invalid", "failed", "not found", "unavailable", 
                       "cannot", "unable", "problem", "issue", "wrong", "fail", "rejected"]
    has_error_next = any(kw in next_state_lower for kw in error_indicators)
    
    if has_error_next:
        safety_score -= 0.45
    else:
        # No errors detected - slight positive for clean state
        safety_score += 0.03
    
    # Statefulness: reward staying in relevant context
    form_words = ["form", "field", "input", "text", "email", "date", "time", "task", "item", "entry", "message", "calendar"]
    form_count_state = sum(1 for w in form_words if w in state_lower)
    form_count_next = sum(1 for w in form_words if w in next_state_lower)
    
    if form_count_next >= form_count_state:
        statefulness_score += 0.08
    elif form_count_next < form_count_state:
        statefulness_score -= 0.12
    
    # Efficiency bonus for purposeful actions
    purposeful_actions = ["fill", "click", "press"]
    if any(a in action_lower for a in purposeful_actions):
        efficiency_score += 0.08
    
    # Normalize and combine scores
    total = progress_score + safety_score + efficiency_score + statefulness_score + completion_bonus
    
    return total, {
        "progress_score": progress_score,
        "safety_score": safety_score,
        "efficiency_score": efficiency_score,
        "statefulness_score": statefulness_score,
        "completion_bonus": completion_bonus,
    }