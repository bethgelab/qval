import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment.
    The value is based on progress markers: identifying the app, 
    reaching an action-oriented page, matching goal-specific entities,
    and the presence of a final 'Submit/Save' button.
    """
    state_lower = state.lower()
    
    # 1. Extract Goal and Content
    # We separate the goal from the observation/accessibility tree to avoid 
    # matching goal keywords within the goal description itself.
    goal = ""
    content = state_lower
    
    # Regular expression to isolate the goal section
    goal_match = re.search(r"goal:\s*(.*?)(?=\n\s*(?:observation|state|accessibility tree):|$)", state_lower, re.DOTALL)
    if goal_match:
        goal = goal_match.group(1).strip()
        # Content is everything following the goal section
        content = state_lower[goal_match.end():]
    
    # 2. Success Detection
    # If a success message is found in the content, the state is nearly the terminal success state.
    success_messages = ["successfully", "completed", "has been created", "has been sent", "added successfully", "saved successfully"]
    for msg in success_messages:
        if msg in content:
            return 0.98

    # 3. App Detection
    # identify which synthetic app the agent is interacting with.
    app_score = 0.0
    apps = {
        "todo": ["todo", "task"],
        "calendar": ["calendar", "event", "schedule"],
        "messenger": ["messenger", "chat", "message"],
        "maps": ["maps", "location", "navigation", "route"],
        "editor": ["editor", "code", "file", "script"]
    }
    current_app = None
    for app, markers in apps.items():
        if any(m in content for m in markers):
            app_score = 0.15
            current_app = app
            break
    
    # 4. Action Page Detection
    # Identify if the agent is on a "creation" or "editing" page (e.g., "New Event").
    action_page_score = 0.0
    action_markers = {
        "todo": ["new task", "add task", "create task"],
        "calendar": ["new event", "add event", "create event"],
        "messenger": ["new message", "compose", "send message"],
        "maps": ["search", "directions", "find"],
        "editor": ["new file", "create file", "save as"]
    }
    if current_app and any(m in content for m in action_markers[current_app]):
        action_page_score = 0.25
    elif any(any(m in content for m in markers) for markers in action_markers.values()):
        # An action page was found, but the specific app wasn't cleanly identified
        action_page_score = 0.15

    # 5. Entity Matching
    # Extract key nouns/entities from the goal and check if they appear on the page.
    entity_score = 0.0
    if goal:
        stop_words = {"this", "that", "with", "from", "then", "here", "there", "the", "and", "for", "your", "some"}
        goal_verbs = {"add", "create", "send", "set", "make", "find", "open", "write", "click", "press", "go"}
        app_all_markers = [m for markers in apps.values() for m in markers]
        
        # We look for words > 2 chars that aren't stop words, verbs, or general app markers
        entities = [w for w in re.findall(r'\w{3,}', goal) 
                    if w not in stop_words and w not in goal_verbs and w not in app_all_markers]
        
        if entities:
            matches = sum(1 for e in entities if e in content)
            entity_score = (matches / len(entities)) * 0.3

    # 6. Final Button Detection
    # Detection of buttons that usually trigger the terminal success state.
    final_button_score = 0.0
    final_btns = ["save", "submit", "send", "create", "confirm", "add", "ok", "done"]
    # Look for markers common in accessibility trees (e.g., [button] Save)
    for btn in final_btns:
        if re.search(rf"\[?button\]?\s*{btn}", content) or re.search(rf"<{btn}", content) or f"button {btn}" in content:
            final_button_score = 0.2
            break

    # Final Value Aggregation
    # Base (0.1) + App (0.15) + ActionPage (0.25) + Entities (0.3) + FinalButton (0.2) = 1.0
    value = 0.1 + app_score + action_page_score + entity_score + final_button_score
    
    return min(value, 0.99)