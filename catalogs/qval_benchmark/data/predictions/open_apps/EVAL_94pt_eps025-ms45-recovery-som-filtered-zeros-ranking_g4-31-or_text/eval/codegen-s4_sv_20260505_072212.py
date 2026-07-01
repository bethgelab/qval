import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    V(s) is approximated based on the progress towards the goal, 
    the current page context, and indicators of completion.
    """
    state_lower = state.lower()
    
    # 1. Extract the goal to understand the context
    # Usually, the goal is stated at the beginning or explicitly labeled.
    goal = ""
    goal_match = re.search(r"goal:\s*(.*)", state_lower)
    if goal_match:
        goal = goal_match.group(1)
    else:
        # If no explicit label, we look for common goal keywords in the first few lines
        lines = state.split('\n')
        if lines:
            goal = lines[0].lower()

    # Map goals to their respective applications
    app_keywords = {
        "calendar": ["calendar", "event", "date", "time", "schedule"],
        "messenger": ["message", "chat", "send", "messenger", "conversation"],
        "todo": ["todo", "task", "list", "checklist", "complete"],
        "maps": ["map", "location", "directions", "search", "address"],
        "code editor": ["code", "editor", "file", "save", "script", "programming"]
    }
    
    target_app = None
    for app, keywords in app_keywords.items():
        if any(k in goal for k in keywords):
            target_app = app
            break
            
    # 2. Identify "Success" indicators (high probability of goal achievement)
    success_indicators = [
        "successfully", "created", "added", "sent", "saved", 
        "completed", "confirmed", "goal achieved", "done"
    ]
    # Only count as success if it's not just the goal description repeating
    # We check for these keywords in the accessibility tree/page content part
    content_part = state_lower
    if "goal:" in state_lower:
        content_part = state_lower.split("goal:")[1] if "goal:" in state_lower else state_lower
        
    if any(indicator in content_part for indicator in success_indicators):
        # We don't return 1.0 because the binary reward is only given after API verification,
        # but we're very confident.
        return 0.95

    # 3. Analyze the current page and elements to determine progress
    # Common elements indicating different stages of the workflow:
    # Stage 3: Final action button is visible and likely interactable.
    # Stage 2: On the correct form page but still filling data.
    # Stage 1: On the correct application landing page.
    # Stage 0: Wrong page or initial state.
    
    final_action_buttons = ["submit", "save", "send", "create", "add event", "confirm"]
    form_indicators = ["input", "text area", "dropdown", "select", "fill"]
    
    # Check if we are on the target app's page
    on_target_app = False
    if target_app:
        if any(k in content_part for k in app_keywords[target_app]):
            on_target_app = True
    else:
        # If we couldn't identify a target app, any app is potentially the target
        if any(app in content_part for app in app_keywords.keys()):
            on_target_app = True

    if not on_target_app:
        return 0.1  # Not on the target app or unknown location

    # Check for final action buttons
    has_final_button = any(btn in content_part for btn in final_action_buttons)
    # Check for form fields
    has_form = any(f in content_part for f in form_indicators)
    
    if has_final_button and has_form:
        return 0.8  # Very close to finishing
    if has_form:
        return 0.5  # In the middle of the task
    if on_target_app:
        return 0.3  # On the right app, but not yet at the final step
        
    return 0.2