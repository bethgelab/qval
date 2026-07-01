import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    The value is based on how closely the current state aligns with a 
    detected goal and the progress toward completion (e.g., navigating 
    to the right page, filling forms, or reaching a success message).
    """
    if not state:
        return 0.0

    # 1. Extract Goal
    # OpenApps states usually contain the goal description.
    goal_match = re.search(r"Goal:\s*(.*)", state, re.IGNORECASE)
    goal_text = goal_match.group(1).lower() if goal_match else ""

    # 2. Success Detection
    # High value for clear indicators of task completion.
    success_keywords = [
        "successfully", "task completed", "confirmed", "saved successfully", 
        "message sent", "event created", "item added", "done", "success"
    ]
    if any(kw in state.lower() for kw in success_keywords):
        return 1.0

    # 3. Error Detection
    # Lower value if we are in an error state.
    error_keywords = ["error", "invalid", "required field", "failed", "not found"]
    has_error = any(kw in state.lower() for kw in error_keywords)

    # 4. Context/Page Alignment
    # Check if the current page/view matches the app mentioned in the goal.
    app_mappings = {
        "todo": ["todo", "task", "list"],
        "calendar": ["calendar", "event", "date", "schedule"],
        "messenger": ["message", "chat", "messenger", "inbox", "conversation"],
        "maps": ["map", "location", "route", "search", "address"],
        "code editor": ["editor", "code", "file", "script", "save file"]
    }
    
    current_app_score = 0.0
    for app, keywords in app_mappings.items():
        if app in goal_text:
            # If the goal is about this app, check if current state contains app keywords
            if any(kw in state.lower() for kw in keywords):
                current_app_score = 0.3
                break

    # 5. Progress/Actionability Detection
    # Check for "Action" elements that suggest we are in the final stages of a task.
    # e.g., "Save", "Submit", "Add", "Send", "Create".
    action_keywords = ["save", "submit", "add", "send", "create", "confirm", "post"]
    # Only count action buttons as positive if we are in the right app context.
    action_score = 0.0
    if current_app_score > 0:
        if any(kw in state.lower() for kw in action_keywords):
            action_score = 0.4
        
    # 6. Specific Keyword Overlap
    # Increase value if specific nouns from the goal appear in the state.
    # Example: Goal "Add task 'Buy Milk'" -> 'Buy Milk' in state.
    overlap_score = 0.0
    if goal_text:
        # Remove common filler words to find meaningful keywords
        fillers = {"a", "an", "the", "to", "in", "on", "at", "of", "add", "create", "send", "set", "update"}
        goal_words = [w for w in re.findall(r'\w+', goal_text) if w not in fillers]
        
        matches = 0
        for word in goal_words:
            if word in state.lower():
                matches += 1
        
        if goal_words:
            overlap_score = (matches / len(goal_words)) * 0.3

    # Total value calculation
    # Base value for starting
    value = 0.1 + current_app_score + action_score + overlap_score
    
    if has_error:
        value *= 0.5

    # Cap and return
    return float(min(max(value, 0.0), 0.95))