import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    The function uses heuristics to determine if the action moved the agent closer to the goal.
    """
    
    # Success indicators: Keywords that often appear when a task is completed
    success_keywords = [
        "success", "created", "added", "sent", "saved", 
        "confirmed", "completed", "done", "finished"
    ]
    
    # Error indicators: Keywords that suggest a mistake or failure
    error_keywords = [
        "error", "invalid", "failed", "incorrect", "wrong", "not found", "404"
    ]
    
    # Action types and high-value targets
    # We look for clicks on buttons that trigger final submissions or additions
    goal_oriented_buttons = ["submit", "save", "add", "send", "create", "confirm", "ok"]

    # 1. Check for immediate success in the next state
    next_state_lower = next_state.lower()
    for kw in success_keywords:
        if kw in next_state_lower:
            # If the action led directly to a success state, it's very high value
            return 0.95

    # 2. Check for immediate failure/errors in the next state
    for kw in error_keywords:
        if kw in next_state_lower:
            return 0.05

    # 3. Analyze the action taken
    # Extract action type and arguments
    action_match = re.match(r"(\w+)\((.*)\)", action)
    if not action_match:
        return 0.2
    
    action_type = action_match.group(1)
    action_args = action_match.group(2)

    # Heuristic: Filling forms is usually progress toward a goal
    if action_type == "fill":
        # If the text filled actually appears in the next state, it's likely successful progress
        # We split by quotes to find the text being filled
        text_parts = re.findall(r"'(.*?)'", action_args)
        if len(text_parts) >= 2:
            fill_text = text_parts[1]
            if fill_text in next_state:
                return 0.7
        return 0.6

    # Heuristic: Clicking "Save/Submit/Add" buttons is high value
    if action_type == "click":
        # We don't have the mapping of bid to text in the action itself,
        # but we can check if the next state changes significantly or if the bid 
        # associated with a goal-oriented button was likely the target.
        # Since we can't see the bid mapping here, we rely on the state transition.
        if state == next_state:
            return 0.1 # No change, likely an ineffective click
        
        # Check if the click happened and the next state looks more "complete" 
        # or moved to a confirmation-like screen.
        return 0.5

    # Heuristic: Pressing keys (like Enter) can be as good as clicking Submit
    if action_type == "press":
        # If the key is 'Enter', it's often a submission
        if "'enter'" in action_args.lower():
            return 0.6
        return 0.4

    # Heuristic: Scrolling and noops are low value unless they reveal new elements
    if action_type == "scroll":
        if state != next_state:
            return 0.3 # Made some movement
        return 0.1
    
    if action_type == "noop":
        return 0.1

    # Default baseline value
    return 0.2