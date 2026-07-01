import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    The Q-value represents the estimated probability of reaching the goal from the given state 
    after taking the action.
    """
    # Keywords that strongly suggest the task has been completed or is nearly complete
    success_keywords = [
        "successfully", "saved", "created", "sent", "confirmed", 
        "done", "completed", "success", "finished", "verified"
    ]
    
    # Convert next_state to lowercase for case-insensitive matching
    next_state_lower = next_state.lower()
    if any(kw in next_state_lower for kw in success_keywords):
        return 0.95

    # Parse the BrowserGym action: e.g., click('12'), fill('13', 'Hello'), press('14', 'Enter')
    # Group 1: action name, Group 2: bid, Group 3: optional value (e.g., fill text)
    action_match = re.match(r"(\w+)\(['\"]([^'\"]+)(?:,\s*['\"]([^'\"]+)['\"])?\)", action)
    if not action_match:
        return 0.1

    act_type = action_match.group(1)
    bid = action_match.group(2)
    
    # Attempt to identify the text associated with the bid in the accessibility tree
    # Bids are typically presented as [bid] "text" or "text" [bid]
    element_text = ""
    # Pattern 1: [12] "Save"
    p1 = re.search(rf"\[{bid}\].*?\"([^\"]+)\"", state)
    if p1:
        element_text = p1.group(1)
    else:
        # Pattern 2: "Save" [12]
        p2 = re.search(rf"\"([^\"]+)\".*?\[{bid}\]", state)
        if p2:
            element_text = p2.group(1)
    
    element_text_lower = element_text.lower()
    
    # Keywords identifying actions that typically move the agent toward a goal
    positive_keywords = [
        "save", "submit", "add", "create", "send", "confirm", 
        "ok", "yes", "post", "next", "done", "new", "enter"
    ]
    # Keywords identifying actions that might undo progress or navigate away
    negative_keywords = [
        "cancel", "back", "delete", "discard", "remove", 
        "reset", "clear", "exit", "close"
    ]

    if act_type == "click":
        if any(kw in element_text_lower for kw in positive_keywords):
            return 0.8
        if any(kw in element_text_lower for kw in negative_keywords):
            return 0.2
        # Generic navigation clicks are neutral-positive
        return 0.4
    
    elif act_type == "fill":
        # Filling a form is typically a required step toward submission
        return 0.6
    
    elif act_type == "press":
        # Pressing the Enter key is often equivalent to clicking a submit button
        if bid.lower() == "enter":
            return 0.7
        return 0.3
        
    elif act_type == "scroll":
        # Scrolling is often necessary but doesn't directly complete the task
        return 0.2
        
    elif act_type == "noop":
        # No-op is generally not an optimal progression action
        return 0.0
        
    # Default value for unrecognized actions
    return 0.1