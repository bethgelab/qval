def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Direct goal achievement signals - return maximum value
    goal_signals = [
        "goal achieved", "task completed", "success", "completed successfully",
        "event created", "message sent", "todo added", "item created",
        "appointment saved", "task finished", "operation complete"
    ]
    if any(signal in state_lower for signal in goal_signals):
        return 1.0
    
    # Base score from progress indicators
    score = 0.0
    
    # Check for completion action buttons (strong signal of being close)
    completion_buttons = ["submit", "save", "send", "add", "create", "done", "confirm", "finish"]
    if any(btn in state_lower for btn in completion_buttons):
        score += 0.25
    
    # Check for form field activity
    form_indicators = ["filled", "entered", "input", "value=", "placeholder", "textbox", "field"]
    if any(ind in state_lower for ind in form_indicators):
        score += 0.2
    
    # Check for task-specific content presence
    task_content = ["calendar", "event", "message", "todo", "task", "appointment", "chat"]
    if any(content in state_lower for content in task_content):
        score += 0.15
    
    # Check for bid/interactive element presence (means we can take actions)
    if "bid" in state_lower or "button" in state_lower or "clickable" in state_lower:
        score += 0.15
    
    # Check for navigation state (being on correct page)
    nav_indicators = ["page", "view", "screen", "tab", "section"]
    if any(nav in state_lower for nav in nav_indicators):
        score += 0.1
    
    # Check for list/item presence (indicates task context)
    list_indicators = ["list", "item", "entry", "record", "row"]
    if any(item in state_lower for item in list_indicators):
        score += 0.1
    
    # Penalize error/failure states
    error_indicators = ["error", "failed", "invalid", "missing", "required", "cannot", "unable"]
    if any(err in state_lower for err in error_indicators):
        score -= 0.2
    
    # Penalize empty/blank states
    if "empty" in state_lower or "blank" in state_lower or "no items" in state_lower:
        score -= 0.15
    
    # Ensure value stays in valid range
    return max(0.0, min(1.0, score))