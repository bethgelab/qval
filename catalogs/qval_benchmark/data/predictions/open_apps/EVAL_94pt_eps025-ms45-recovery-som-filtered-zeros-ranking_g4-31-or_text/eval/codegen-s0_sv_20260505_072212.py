import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment state.
    The value is based on heuristics derived from the accessibility tree representation,
    identifying markers of goal completion, proximity to goal, and general progress.
    """
    if not state:
        return 0.0

    # Convert state to lowercase for easier matching
    state_lower = state.lower()

    # 1. Goal Achievement / Success signals
    # These keywords usually appear in a confirmation message or toast after a successful action.
    success_patterns = [
        r"successfully", 
        r"created", 
        r"saved", 
        r"sent", 
        r"updated", 
        r"completed", 
        r"task added", 
        r"event added", 
        r"message sent"
    ]
    for pattern in success_patterns:
        if re.search(pattern, state_lower):
            # High value, but not necessarily 1.0 unless we are certain it's the final terminal state.
            # In binary reward environments, we approximate based on perceived completion.
            return 0.95

    # 2. Proximity to Goal / Final Action signals
    # Presence of primary action buttons like 'Save', 'Submit', 'Send', 'Create'
    # suggests the agent has filled the form and is one step away from success.
    final_action_patterns = [
        r"button.*save", 
        r"button.*submit", 
        r"button.*send", 
        r"button.*create", 
        r"button.*confirm",
        r"button.*add"
    ]
    # Check if any of these buttons are present and likely visible/interactive
    for pattern in final_action_patterns:
        if re.search(pattern, state_lower):
            return 0.75

    # 3. Progress / Form Interaction signals
    # If we are on a 'New' or 'Add' page, or there are multiple input fields present,
    # the agent is making progress toward the goal.
    progress_patterns = [
        r"new task", 
        r"add event", 
        r"compose message", 
        r"create item", 
        r"textbox", 
        r"input", 
        r"fill"
    ]
    progress_count = 0
    for pattern in progress_patterns:
        if re.search(pattern, state_lower):
            progress_count += 1
            
    if progress_count >= 2:
        return 0.5
    elif progress_count == 1:
        return 0.3

    # 4. Navigation / Starting Point signals
    # Being on the main dashboard or home page is better than being in a lost/error state.
    home_patterns = [
        r"dashboard", 
        r"home", 
        r"todo list", 
        r"calendar view", 
        r"messages", 
        r"map view", 
        r"editor"
    ]
    for pattern in home_patterns:
        if re.search(pattern, state_lower):
            return 0.15

    # 5. Negative signals / Error states
    # Error messages decrease the value of the state.
    error_patterns = [
        r"error", 
        r"invalid", 
        r"required", 
        r"failed", 
        r"not found"
    ]
    for pattern in error_patterns:
        if re.search(pattern, state_lower):
            return 0.05

    # Default base value for any other state
    return 0.1