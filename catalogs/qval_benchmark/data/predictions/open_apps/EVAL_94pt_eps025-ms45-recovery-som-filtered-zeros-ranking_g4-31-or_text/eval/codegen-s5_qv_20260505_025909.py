import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an OpenApps environment.
    """
    # 1. Identify the target element text if applicable
    target_text = ""
    # Extract bid from action (e.g., click('12'), fill('5', 'text'))
    bid_match = re.search(r"'(.*?)'", action)
    if bid_match:
        bid = bid_match.group(1)
        # Try to find the text associated with this bid in the state representation
        # Assuming accessibility tree format where bid is associated with element text
        # Look for the bid and capture subsequent text until the next element or newline
        bid_pattern = re.compile(rf"bid={bid}.*?>(.*?)<", re.IGNORECASE | re.DOTALL)
        match = bid_pattern.search(state)
        if match:
            target_text = match.group(1).strip().lower()

    # 2. Check for Goal Achievement (High Q-value)
    # Success markers typically appear in the next_state after the final action
    success_markers = [
        "successfully", "created", "saved", "sent", "added", 
        "completed", "deleted", "updated", "confirmed"
    ]
    
    # If next_state contains a success marker that state didn't
    for marker in success_markers:
        if marker in next_state.lower() and marker not in state.lower():
            return 1.0

    # 3. Evaluate Action based on type and target
    # Actions that move the agent closer to the goal
    progress_markers = ["save", "submit", "create", "add", "send", "ok", "confirm", "new"]
    regress_markers = ["cancel", "back", "close", "delete", "remove", "discard"]

    # High value for clicking "Submit/Save" etc.
    if "click" in action:
        if any(marker in target_text for marker in progress_markers):
            return 0.8
        if any(marker in target_text for marker in regress_markers):
            return 0.1
        return 0.3

    # Mid value for filling out form fields
    if "fill" in action:
        # Filling a form is usually progress unless it's a "cancel" or "delete" field
        return 0.5

    # Low value for navigation or scrolling
    if "scroll" in action or "noop" in action:
        # Noop is generally inefficient unless the page is loading
        return 0.05 if "noop" in action else 0.2

    # 4. Heuristic state transition analysis
    # Transition from a "list" page to a "form" page is progress
    form_markers = ["create", "new", "edit", "add"]
    state_has_form = any(m in state.lower() for m in form_markers)
    next_has_form = any(m in next_state.lower() for m in form_markers)
    
    if not state_has_form and next_has_form:
        return 0.6
    if state_has_form and not next_has_form:
        # Could be a submission (covered by success_markers) or a cancellation
        return 0.2

    # Baseline Q-value for unspecified actions
    return 0.1