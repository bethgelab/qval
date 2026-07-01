import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    V(s) is approximated by identifying the progress towards goal completion
    based on keywords in the accessibility tree and the task description.
    """
    # 1. Try to extract the goal from the state string
    # Common patterns for goal specification in synthetic environments
    goal_match = re.search(r"(?:Goal|Task|Instruction):\s*(.*)", state, re.IGNORECASE)
    goal_text = goal_match.group(1).strip().lower() if goal_match else ""
    
    # 2. Success Detection (Highest Value)
    # Keywords that indicate the task has been completed.
    success_keywords = [
        "successfully", "completed", "created", "sent", "added", 
        "saved", "confirmed", "done", "success", "finished"
    ]
    # We check for these words, especially if they are in a context suggesting a notification or header
    for word in success_keywords:
        # Match word as a standalone token to avoid partial matches
        if re.search(rf"\b{word}\b", state, re.IGNORECASE):
            # To avoid false positives (e.g., "Add a created item"), we prioritize 
            # markers of finality or completion messages.
            if any(marker in state.lower() for marker in ["success", "confirmed", "saved successfully"]):
                return 1.0
            # If it's a generic success word, we give it a very high value but not necessarily 1.0
            # unless it's clearly the outcome.
            return 0.95

    # 3. Near-Completion Detection (High Value)
    # Elements that are usually the final step of a task (Submit buttons).
    final_action_keywords = [
        "save", "submit", "create", "send", "add", "confirm", "post", "update"
    ]
    # Check for buttons with these labels
    for word in final_action_keywords:
        # Look for common browser-gym/accessibility patterns like [button "Save"]
        if re.search(rf'button\s*["\']{word}["\']', state, re.IGNORECASE) or \
           re.search(rf'["\']{word}["\']\s*button', state, re.IGNORECASE) or \
           re.search(rf'\b{word}\b', state, re.IGNORECASE):
            # If the goal specifically mentions this action, it's higher value
            if goal_text and word in goal_text:
                return 0.85
            return 0.75

    # 4. Progress Detection (Medium Value)
    # Being on the correct form or page.
    # Check for input fields or form elements.
    input_indicators = ["input", "textarea", "select", "fill", "type"]
    has_inputs = any(ind in state.lower() for ind in input_indicators)
    
    # Check if the current page context matches the goal
    app_keywords = {
        "todo": ["todo", "task", "list", "checklist"],
        "calendar": ["calendar", "event", "date", "schedule", "appointment"],
        "messenger": ["message", "chat", "messenger", "conversation", "dm"],
        "maps": ["map", "location", "search", "directions", "place"],
        "code editor": ["code", "editor", "file", "script", "save file", "text editor"]
    }
    
    on_correct_app = False
    if goal_text:
        for app, keys in app_keywords.items():
            if any(k in goal_text for k in keys) and any(k in state.lower() for k in keys):
                on_correct_app = True
                break
    
    if has_inputs and on_correct_app:
        return 0.6
    if on_correct_app:
        return 0.4
    if has_inputs:
        return 0.3

    # 5. Initial/Low Progress (Low Value)
    # Generic navigation state or home page.
    if "home" in state.lower() or "welcome" in state.lower():
        return 0.2

    return 0.1