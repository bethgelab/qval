import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    """
    # Success and failure keywords to detect transitions to target or error states
    success_keywords = ["successfully", "confirmed", "created", "added", "sent", "saved", "completed", "finished", "done"]
    error_keywords = ["error", "failed", "invalid", "incorrect", "denied", "wrong", "not found"]
    
    # High-value action labels (likely to complete a task)
    high_value_labels = ["save", "submit", "send", "confirm", "ok", "create", "finish"]
    # Mid-value action labels (likely to make progress)
    mid_value_labels = ["add", "new", "compose", "edit", "plus"]
    # Low-value action labels (likely to regress or stay neutral)
    low_value_labels = ["cancel", "back", "delete", "clear", "remove"]

    # Check for immediate goal achievement markers in next_state that were not in state
    success_found = any(word in next_state.lower() for word in success_keywords)
    success_already_present = any(word in state.lower() for word in success_keywords)
    
    if success_found and not success_already_present:
        return 1.0

    # Check for error markers in next_state that were not in state
    error_found = any(word in next_state.lower() for word in error_keywords)
    error_already_present = any(word in state.lower() for word in error_keywords)
    
    if error_found and not error_already_present:
        return 0.0

    # Analyze action and state transition
    score = 0.3  # Default base value
    
    # Extract bid and action type
    # Common patterns: click('12'), fill('13', 'text'), press('14', 'Enter')
    action_type = ""
    bid = None
    
    if "click(" in action:
        action_type = "click"
        match = re.search(r"click\('(\d+)'\)", action)
        if match:
            bid = match.group(1)
    elif "fill(" in action:
        action_type = "fill"
        match = re.search(r"fill\('(\d+)',", action)
        if match:
            bid = match.group(1)
    elif "press(" in action:
        action_type = "press"
        match = re.search(r"press\('(\d+)',", action)
        if match:
            bid = match.group(1)
    elif "scroll" in action:
        action_type = "scroll"
    elif "noop" in action:
        action_type = "noop"

    # If we have a bid, look for the associated element text in the accessibility tree (state)
    element_text = ""
    if bid:
        # Bids are typically represented as [bid] in the accessibility tree text
        pattern = rf"\[{bid}\].*?\"(.*?)\""
        match = re.search(pattern, state)
        if match:
            element_text = match.group(1).lower()
        else:
            # Fallback: just search for the bid in the state string
            lines = state.split('\n')
            for line in lines:
                if f"[{bid}]" in line:
                    element_text = line.lower()
                    break

    # Heuristic scoring based on action and element text
    if action_type == "click":
        if any(word in element_text for word in high_value_labels):
            score = 0.8
        elif any(word in element_text for word in mid_value_labels):
            score = 0.6
        elif any(word in element_text for word in low_value_labels):
            score = 0.2
        else:
            score = 0.4
    elif action_type == "fill":
        # Filling a form is generally a sign of progress
        score = 0.5
    elif action_type == "press":
        # Pressing 'Enter' is often similar to clicking 'Submit'
        if "Enter" in action:
            score = 0.7
        else:
            score = 0.4
    elif action_type == "scroll":
        score = 0.3
    elif action_type == "noop":
        score = 0.1

    # Penalty if the action resulted in no change to the state
    if state == next_state:
        score -= 0.2

    # Ensure final score is within [0.0, 1.0]
    return max(0.0, min(1.0, score))