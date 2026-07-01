import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    V(s) represents the expected discounted cumulative reward from the current state.
    Since the reward is binary (1.0 upon goal completion), V(s) is approximated 
    as the probability of reaching the goal from the current state.
    """
    s = state.lower()
    
    # 1. Goal Achievement Markers (Highest Value)
    # These keywords typically appear when the server API verifies the goal has been met
    # or when the UI displays a final success notification.
    success_keywords = [
        "successfully", "goal achieved", "task completed", "was created", 
        "was sent", "was deleted", "was updated", "saved successfully", 
        "done", "confirmed", "completed"
    ]
    if any(kw in s for kw in success_keywords):
        # Further verify it's a confirmation message and not just a button label
        if any(x in s for x in ["success", "done", "confirmed", "achieved"]):
            return 1.0
        return 0.95

    # 2. Final Step Markers (Very High Value)
    # States where the user is on the final confirmation page or about to click the final submit button.
    final_step_keywords = [
        "submit", "save changes", "confirm", "send message", 
        "create event", "create task", "save file", "confirm deletion"
    ]
    if any(kw in s for kw in final_step_keywords):
        # If these buttons are present along with input fields, we are likely at the final step.
        if any(x in s for x in ["input", "text", "field", "placeholder", "value="]):
            return 0.85
        return 0.75

    # 3. Intermediate Progress Markers (Moderate Value)
    # States where the agent has navigated to the correct tool or opened a creation form.
    progress_keywords = [
        "add new", "create", "new event", "new task", "compose", 
        "search for", "edit", "schedule", "destination", "file explorer"
    ]
    if any(kw in s for kw in progress_keywords):
        return 0.5

    # 4. Starting/Generic Navigation (Low Value)
    # Home pages, dashboards, or main menus.
    start_keywords = [
        "home", "dashboard", "main menu", "index", "welcome", 
        "my tasks", "my calendar", "chat list"
    ]
    if any(kw in s for kw in start_keywords):
        return 0.3

    # 5. Negative Markers (Lowest Value)
    # Error messages, invalid inputs, or "not found" pages.
    error_keywords = [
        "error", "invalid", "required", "not found", "failed", 
        "incorrect", "try again", "unexpected"
    ]
    if any(kw in s for kw in error_keywords):
        return 0.1

    # Default value for ambiguous states
    return 0.2