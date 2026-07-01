import math

def signal_function(state: str) -> float:
    if not state:
        return 0.0
    
    s = state.lower()
    
    # Success Indicators: Directly implies task completion
    success_patterns = [
        "success", "completed", "created", "added", "saved", "sent", 
        "done", "task complete", "event created", "message sent", 
        "task added", "task completed", "calendar event added",
        "message sent successfully", "todo added", "code saved"
    ]
    for pattern in success_patterns:
        if pattern in s:
            return 1.0
            
    # Error Indicators: Directly implies failure or blockage
    error_patterns = [
        "error", "failed", "invalid", "required", "cannot", "not found",
        "timeout", "connection refused", "permission denied", "unauthorized"
    ]
    for pattern in error_patterns:
        if pattern in s:
            return 0.0
            
    # Intermediate Progress Estimation
    # Base score for a non-terminal state
    score = 0.2
    
    # Actionability: Presence of submit/send/save buttons implies proximity to goal
    action_patterns = ["submit", "send", "save", "create", "add", "post", "confirm"]
    for pattern in action_patterns:
        if pattern in s:
            score += 0.3
            
    # Form Completion: Presence of inputs implies task engagement
    form_patterns = ["textbox", "input", "textarea", "select", "checkbox"]
    for pattern in form_patterns:
        if pattern in s:
            score += 0.1
            
    # Navigation: Presence of links/buttons implies ability to proceed
    nav_patterns = ["button", "link", "href"]
    for pattern in nav_patterns:
        if pattern in s:
            score += 0.1
            
    # Negative Signals: Loading or stuck states reduce value
    stuck_patterns = ["loading", "waiting", "disabled", "processing"]
    for pattern in stuck_patterns:
        if pattern in s:
            score -= 0.2
            
    # Initial State: Home or empty pages are further from goal
    initial_patterns = ["home", "start", "new", "empty", "welcome", "login"]
    for pattern in initial_patterns:
        if pattern in s:
            score -= 0.1
            
    # Ensure value is within [0.0, 1.0]
    return max(0.0, min(1.0, score))