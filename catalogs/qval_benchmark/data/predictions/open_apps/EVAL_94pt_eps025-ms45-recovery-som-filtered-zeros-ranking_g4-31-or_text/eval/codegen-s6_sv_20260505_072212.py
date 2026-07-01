import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment.
    V(s) is the expected discounted cumulative reward starting from state s.
    Reward is binary: 1.0 on goal achievement, 0.0 otherwise.
    """
    # 1. Goal Extraction
    # The state string usually contains the goal at the beginning or under a label.
    goal = ""
    goal_match = re.search(r"(?:Goal|Task|Objective):\s*(.*)", state, re.IGNORECASE)
    if goal_match:
        goal = goal_match.group(1).lower()
    else:
        # Fallback: use the first line as the goal if no explicit label is found
        goal = state.split('\n')[0].lower()

    tree = state.lower()
    
    # Baseline value for being in the environment
    value = 0.1
    
    # 2. Target Entity Extraction
    # Identify the specific object/text the agent is looking for or creating.
    target_entity = ""
    quoted = re.findall(r"'(.*?)'|\"(.*?)\"", goal)
    if quoted:
        # Take the first non-empty group from the first quoted match
        target_entity = next((item for item in quoted[0] if item), "")
    else:
        # Heuristic: assume the last few words of the goal describe the entity
        words = goal.split()
        if len(words) > 2:
            target_entity = " ".join(words[-2:])
    
    target_entity = target_entity.lower()

    # 3. Terminal State Detection (High Value)
    # Look for success confirmations.
    success_indicators = ["successfully", "confirmed", "saved", "sent", "completed"]
    for ind in success_indicators:
        if ind in tree and any(kw in goal for kw in ["create", "add", "send", "update", "delete"]):
            return 0.95

    # If the target entity is present in the current view
    if target_entity and target_entity in tree:
        # Distinguish between "filling a form" and "seeing the result"
        # Form indicators: input fields, "Save" or "Submit" buttons still present.
        form_buttons = ["save", "submit", "create", "add", "ok", "confirm"]
        is_in_form = any(btn in tree for btn in form_buttons)
        
        if not is_in_form:
            # Entity is visible and no submit button is present -> likely achieved.
            return 0.98
        else:
            # Entity is visible but we are still in a form -> nearly done.
            return 0.85

    # 4. Progress Estimation
    # Action: Create/Add/New
    if any(kw in goal for kw in ["create", "add", "new"]):
        # Check if we have navigated to the creation page
        if any(kw in tree for kw in ["new", "create", "add", "form", "plus", "insert"]):
            # If we are on the page and a submit button is visible, we are halfway.
            if any(btn in tree for btn in ["save", "submit", "create", "add"]):
                value = max(value, 0.6)
            else:
                value = max(value, 0.4)

    # Action: Send/Message
    if "send" in goal or "message" in goal:
        if any(kw in tree for kw in ["chat", "message", "conversation", "messenger", "contact"]):
            if "send" in tree:
                value = max(value, 0.7)
            else:
                value = max(value, 0.4)

    # Action: Navigate/Find/Search/Open
    if any(kw in goal for kw in ["navigate", "find", "search", "go to", "open", "view"]):
        # Check for keyword overlap between the goal and the current accessibility tree.
        goal_words = [w for w in goal.split() if len(w) > 3]
        matches = sum(1 for w in goal_words if w in tree)
        if matches >= 2:
            value = max(value, 0.7)
        elif matches >= 1:
            value = max(value, 0.4)

    # 5. App-Specific Context
    # Check if the agent is at least in the correct application.
    app_contexts = {
        "todo": ["todo", "task", "checklist"],
        "calendar": ["calendar", "event", "date", "time"],
        "messenger": ["messenger", "chat", "message"],
        "maps": ["maps", "location", "place", "search"],
        "code editor": ["editor", "code", "file", "script", "text edit"]
    }
    
    for app_name, keywords in app_contexts.items():
        if any(kw in goal for kw in keywords) and any(kw in tree for kw in keywords):
            value = max(value, 0.2)

    return float(min(value, 1.0))