import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment.
    The value represents the expected discounted cumulative reward,
    which in this binary reward setting is essentially the probability 
    of reaching the goal from the current state.
    """
    if not state:
        return 0.0

    state_lower = state.lower()
    
    # 1. Extract Goal
    # The state typically contains a "Goal: ..." section.
    goal_match = re.search(r"Goal:\s*(.*)", state, re.IGNORECASE)
    goal_text = goal_match.group(1).lower() if goal_match else state_lower

    # 2. Identify the target application from the goal
    apps = ["todo", "calendar", "messenger", "map", "editor"]
    target_app = None
    for app in apps:
        if app in goal_text:
            target_app = app
            break

    # 3. Define heuristic markers for progression
    success_markers = ["successfully", "completed", "created", "sent", "updated", "deleted", "done", "confirmed"]
    submit_markers = ["save", "submit", "send", "confirm", "ok", "create", "add", "post", "search"]
    form_markers = ["input", "textbox", "field", "type", "fill", "edit", "write"]

    # 4. State-Value Estimation Logic (Hierarchical)
    
    # Level 1: Goal likely achieved (Value = 1.0)
    # We look for a success marker combined with a reference to the app or the goal content.
    if any(marker in state_lower for marker in success_markers):
        # To avoid false positives on home pages, ensure we are in the right app context.
        if target_app and target_app in state_lower:
            return 1.0
        if not target_app and any(marker in state_lower for marker in success_markers):
            return 1.0

    # Level 2: Final action step (Value = 0.8)
    # We are on the correct page and a submission button is visible.
    # Look for markers that appear in the context of a button.
    if "button" in state_lower or 'role="button"' in state_lower:
        if any(marker in state_lower for marker in submit_markers):
            # Ensure we are in the target app context to avoid valuing "Save" buttons on the wrong page.
            if target_app and target_app in state_lower:
                return 0.8
            if not target_app:
                return 0.8

    # Level 3: Intermediate progress - filling a form (Value = 0.5)
    # We are interacting with inputs or a creation screen.
    if any(marker in state_lower for marker in form_markers):
        if target_app and target_app in state_lower:
            return 0.5
        if not target_app:
            return 0.5

    # Level 4: Correct application reached (Value = 0.2)
    # We have navigated to the app mentioned in the goal.
    if target_app and target_app in state_lower:
        return 0.2

    # Level 5: Search-specific progress
    # If the goal is a search, finding the search term in the results is highly valuable.
    if "search" in goal_text:
        # Try to extract the search term (simple heuristic: word after 'for' or 'of')
        search_term_match = re.search(r"search for\s+([\w\s]+?)(?: to| in| on|$)", goal_text)
        if search_term_match:
            term = search_term_match.group(1).strip()
            if term and term in state_lower:
                return 0.8

    # Base value for being at the start or on an irrelevant page.
    return 0.0