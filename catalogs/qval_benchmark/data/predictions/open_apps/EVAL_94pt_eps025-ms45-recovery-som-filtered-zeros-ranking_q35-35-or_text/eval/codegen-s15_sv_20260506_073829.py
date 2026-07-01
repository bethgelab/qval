import re

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # 1. Check for explicit goal completion (Value = 1.0)
    success_patterns = [
        "success", "completed", "saved", "created", "added", "sent", 
        "done", "event added", "message sent", "task completed", 
        "file saved", "item added", "saved successfully", "sent successfully",
        "operation successful", "request completed"
    ]
    for pattern in success_patterns:
        if pattern in state_lower:
            return 1.0

    # 2. Check for errors or dead ends (Value = 0.0)
    error_patterns = [
        "error", "failed", "404", "not found", "invalid", 
        "unauthorized", "session expired", "timeout"
    ]
    for pattern in error_patterns:
        if pattern in state_lower:
            return 0.0

    # 3. Estimate progress based on UI elements
    # High value: Final action button visible (Save, Send, Submit, Add)
    action_patterns = ["save", "submit", "send", "confirm", "add", "create", "done"]
    has_action = any(pattern in state_lower for pattern in action_patterns)

    # Medium value: Form fields or list context present
    form_patterns = ["input", "textbox", "textarea", "form", "list", "field"]
    has_form = any(pattern in state_lower for pattern in form_patterns)

    # Low value: Navigation or login screens (Further from goal)
    nav_patterns = ["login", "sign in", "home", "welcome", "select", "main menu", "dashboard"]
    has_nav = any(pattern in state_lower for pattern in nav_patterns)

    # 4. Calculate Value Score
    # Base value for unknown state
    score = 0.1
    
    if has_action and has_form:
        # Ready to submit final action
        score = 0.85
    elif has_action:
        # Action visible but context unclear
        score = 0.65
    elif has_form:
        # Filling form but no action visible yet
        score = 0.45
    elif has_nav:
        # Navigating or logging in
        score = 0.25
    else:
        # Generic state
        score = 0.15

    return score