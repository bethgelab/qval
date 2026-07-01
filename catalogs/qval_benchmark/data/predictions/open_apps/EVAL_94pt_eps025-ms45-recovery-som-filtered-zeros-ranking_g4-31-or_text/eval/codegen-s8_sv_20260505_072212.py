import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment.
    V(s) is the expected discounted cumulative reward (probability of success).
    """
    if not state:
        return 0.0

    state_lower = state.lower()
    
    # 1. Check for explicit goal completion markers (High Value: 0.9 - 1.0)
    # These are common patterns in synthetic apps indicating the task is finished.
    success_patterns = [
        "successfully", 
        "task completed", 
        "message sent", 
        "event created", 
        "item added", 
        "saved successfully", 
        "deleted successfully", 
        "updated successfully",
        "confirmation: success"
    ]
    if any(pattern in state_lower for pattern in success_patterns):
        return 1.0

    # 2. Identify the goal to contextualize progress
    # Usually, the goal is provided at the start of the state string.
    goal_match = re.search(r"Goal:\s*(.*?)(\n|$)", state, re.IGNORECASE)
    goal = goal_match.group(1).lower() if goal_match else ""

    # 3. Check for "Final Step" indicators (Medium-High Value: 0.7 - 0.8)
    # Being on the final confirmation button or the 'Submit' page.
    final_actions = ["submit", "save", "send", "confirm", "create", "delete", "update"]
    # We look for these as interactive elements (buttons/links)
    # The accessibility tree usually represents them as [bid=X] Button "Save"
    for action in final_actions:
        # Ensure the action is relevant to the goal (e.g., don't value 'delete' if goal is 'create')
        if action in goal or (not goal and action in state_lower):
            # Look for common patterns of buttons/interactive elements
            if f'button "{action}"' in state_lower or f'link "{action}"' in state_lower or f'role="button"' in state_lower and action in state_lower:
                # If we see a final action button AND form fields, we are very close.
                if any(field in state_lower for field in ["input", "textarea", "select"]):
                    return 0.8
                return 0.7

    # 4. Check for "Intermediary Step" indicators (Medium Value: 0.4 - 0.6)
    # We are on the correct page (e.g., the 'New Event' form) but haven't submitted yet.
    intermediary_patterns = ["new", "add", "create", "compose", "edit"]
    
    # Check if we are on a creation/edit page by looking for input fields
    has_inputs = any(field in state_lower for field in ["input", "textarea", "select", "placeholder"])
    
    # If the goal involves creating/adding and we are on a page with inputs, it's progress.
    if any(p in goal for p in intermediary_patterns) and has_inputs:
        return 0.6
    
    # If we have clicked "New" or "Add" and are moving toward the form
    for p in intermediary_patterns:
        if p in goal and (f'button "{p}"' in state_lower or f'link "{p}"' in state_lower):
            return 0.4

    # 5. Check for "Navigation" progress (Low-Medium Value: 0.2 - 0.3)
    # We are on the right app section (e.g., "Calendar" page for a calendar goal).
    app_keywords = {
        "calendar": ["calendar", "event", "date"],
        "todo": ["todo", "task", "list"],
        "messenger": ["message", "chat", "messenger", "contact"],
        "maps": ["map", "location", "search", "address"],
        "code editor": ["editor", "code", "file", "script"]
    }
    
    for app, keywords in app_keywords.items():
        if any(k in goal for k in keywords):
            if any(k in state_lower for k in keywords):
                return 0.2

    # 6. Base Value (Lowest Value: 0.0 - 0.1)
    # Starting state or irrelevant page.
    return 0.05