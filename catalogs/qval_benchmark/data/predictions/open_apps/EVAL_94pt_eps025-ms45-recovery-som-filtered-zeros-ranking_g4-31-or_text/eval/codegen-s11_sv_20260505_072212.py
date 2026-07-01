import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the OpenApps environment.
    The value reflects the probability of achieving the goal, weighted by efficiency.
    """
    if not state:
        return 0.0

    state_lower = state.lower()
    
    # 1. High-confidence success indicators (Goal achieved or very close)
    # These are patterns often found in confirmation messages or status updates.
    success_patterns = [
        r"successfully", 
        r"has been created", 
        r"has been added", 
        r"message sent", 
        r"saved successfully", 
        r"confirmed", 
        r"done",
        r"goal reached",
        r"task completed"
    ]
    for pattern in success_patterns:
        if re.search(pattern, state_lower):
            return 0.95

    # 2. High-progress indicators (Being on the final action screen)
    # Presence of a submit/save button often means the agent is one step from victory.
    final_action_patterns = [
        r"button.*submit", 
        r"button.*save", 
        r"button.*send", 
        r"button.*create", 
        r"button.*add",
        r"button.*confirm"
    ]
    # Check for buttons that trigger the final state change
    for pattern in final_action_patterns:
        if re.search(pattern, state_lower):
            # Check if inputs are filled to increase confidence
            # Look for value="something" where something is not empty
            if re.search(r'value="[^"]+"', state_lower) or re.search(r'text="[^"]+"', state_lower):
                return 0.8
            return 0.6

    # 3. Mid-progress indicators (Being in the "Create/Edit" flow)
    # These pages are necessary stepping stones but not yet the finish line.
    progress_patterns = [
        r"new event", 
        r"new task", 
        r"compose message", 
        r"create new", 
        r"edit", 
        r"add entry",
        r"form", 
        r"input"
    ]
    for pattern in progress_patterns:
        if re.search(pattern, state_lower):
            # If the user is in a "Create" page, they are significantly closer than the home page
            return 0.4

    # 4. Low-progress indicators (Main dashboards or navigation)
    # Being on the home page is better than an error page, but far from the goal.
    home_patterns = [
        r"dashboard", 
        r"home", 
        r"all tasks", 
        r"calendar view", 
        r"inbox", 
        r"main menu"
    ]
    for pattern in home_patterns:
        if re.search(pattern, state_lower):
            return 0.15

    # Default value for states that don't match specific heuristics
    return 0.05