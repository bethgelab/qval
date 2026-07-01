def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Check for task completion indicators
    completion_patterns = [
        r"success", r"completed", r"task complete", r"goal achieved",
        r"verified", r"saved successfully", r"sent successfully",
        r"created successfully", r"added successfully", r"confirmed"
    ]
    for pattern in completion_patterns:
        if re.search(pattern, state_lower):
            return 1.0
    
    # Estimate progress based on state features
    
    # 1. Count interactive elements (bids indicate available actions)
    bid_matches = len(re.findall(r"bid['\"]?\s*[:=]\s*['\"]?\d+['\"]?", state_lower))
    bid_score = min(1.0, bid_matches / 15.0)
    
    # 2. Count buttons and clickable elements
    button_count = len(re.findall(r"button|clickable|link|click", state_lower))
    button_score = min(1.0, button_count / 10.0)
    
    # 3. Count form inputs (indicates we can fill data)
    input_count = len(re.findall(r"input|text|field|form|textarea", state_lower))
    input_score = min(1.0, input_count / 8.0)
    
    # 4. Check for app-specific navigation indicators
    app_indicators = ["todo", "calendar", "messenger", "maps", "code", "editor", "app"]
    app_score = min(1.0, sum(1 for app in app_indicators if app in state_lower) / 3.0)
    
    # 5. Check for progress/action keywords
    action_keywords = ["add", "create", "send", "save", "submit", "fill", "enter", "click", "open"]
    action_count = sum(1 for kw in action_keywords if kw in state_lower)
    action_score = min(1.0, action_count / 5.0)
    
    # 6. Check for error/negative indicators (reduce value if present)
    error_patterns = [r"error", r"failed", r"invalid", r"not found", r"cannot"]
    error_count = sum(1 for pattern in error_patterns if re.search(pattern, state_lower))
    error_penalty = min(0.3, error_count * 0.1)
    
    # 7. Check for confirmation/progress messages
    progress_messages = [r"step", r"page \d+", r"current", r"view", r"loading"]
    progress_count = sum(1 for pattern in progress_messages if re.search(pattern, state_lower))
    progress_bonus = min(0.2, progress_count * 0.05)
    
    # Combine features into progress estimate
    progress_score = (
        bid_score * 0.20 +
        button_score * 0.20 +
        input_score * 0.15 +
        app_score * 0.15 +
        action_score * 0.20 +
        progress_bonus
    )
    
    # Apply error penalty
    progress_score = max(0.0, progress_score - error_penalty)
    
    # Factor in step efficiency (earlier in episode = more room to succeed)
    # Assume average episode is around 20-30 steps, so mid-episode value ~0.5
    step_efficiency = 0.6
    
    # Final value estimate
    value = progress_score * step_efficiency
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, value))