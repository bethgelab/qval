def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment based on the 
    accessibility tree representation of the current page.
    """
    s = state.lower()
    
    # 1.0: High confidence of goal achievement.
    # We look for common success markers and confirmation messages.
    success_markers = [
        "successfully", "completed", "has been added", 
        "has been sent", "has been created", "has been saved", 
        "task added", "event created", "message sent", 
        "confirmation", "success", "done"
    ]
    if any(marker in s for marker in success_markers):
        return 1.0
        
    # 0.8: Very close - The final action button is likely visible.
    # Buttons that typically trigger the transition to the goal state.
    final_action_markers = [
        "save", "submit", "send", "confirm", 
        "finish", "create event", "add task", "save file"
    ]
    if any(marker in s for marker in final_action_markers):
        # To ensure we are actually at the final step, we check for these
        # keywords. If they appear as buttons, the value is high.
        return 0.8
        
    # 0.5: Mid-way - Interacting with a form or input field.
    # Presence of input elements suggests the agent is in the process of providing data.
    input_markers = [
        "input", "textbox", "placeholder", "type", 
        "select", "dropdown", "date", "time", "fill"
    ]
    if any(marker in s for marker in input_markers):
        return 0.5
        
    # 0.3: Started - Navigated to a 'creation' or 'action' screen.
    # Common buttons that initiate a workflow.
    start_markers = [
        "new", "add", "create", "compose", "plus"
    ]
    if any(marker in s for marker in start_markers):
        return 0.3
        
    # 0.1: Base - The agent is within one of the targeted application contexts.
    app_markers = [
        "calendar", "todo", "messenger", "maps", "editor", "dashboard", "home"
    ]
    if any(marker in s for marker in app_markers):
        return 0.1
        
    # 0.0: No progress markers identified.
    return 0.0