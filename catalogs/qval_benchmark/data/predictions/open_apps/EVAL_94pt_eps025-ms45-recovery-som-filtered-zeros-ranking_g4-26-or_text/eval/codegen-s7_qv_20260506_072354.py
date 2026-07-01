import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a given state, action, and resulting next state.
    Since the reward is binary and outcome-based, the Q-value approximates the 
    probability of the current action leading to the task goal.
    """
    ns_lower = next_state.lower()
    
    # 1. Task Achievement Detection
    # Many OpenApps environments use specific success indicators in the accessibility tree.
    # If these keywords appear, it's highly likely the goal has been achieved.
    success_patterns = [
        "success", "sent", "saved", "added", "created", 
        "completed", "done", "updated", "message sent", 
        "event added", "task added", "appointment added"
    ]
    if any(pattern in ns_lower for pattern in success_patterns):
        return 1.0

    # 2. Failure/Error Detection
    # If the action results in an error message, the Q-value should be low.
    error_patterns = ["error", "failed", "invalid", "not found", "incorrect", "wrong"]
    if any(pattern in ns_lower for pattern in error_patterns):
        return 0.0

    # 3. Heuristic Progress Estimation
    # We analyze the 'action' to see if it's a productive interaction (fill, click, press)
    # and check if the 'next_state' reflects a meaningful change.

    # Case A: 'fill' action
    # We look for the text being entered in the action and check if it appears in the next state.
    # Regex captures the content within the second argument of fill('bid', 'text').
    fill_match = re.search(r"fill\(\s*['\"].*?['\"]\s*,\s*['\"](.*?)['\"]\s*\)", action)
    if fill_match:
        text_val = fill_match.group(1)
        # If the text entered is actually present in the tree, it's a high-quality step.
        if text_val and text_val in next_state:
            return 0.7
        # If it's a fill attempt but the state hasn't updated yet, it's still a positive step.
        return 0.1

    # Case B: 'press' action
    # Pressing "Enter" is a common way to submit forms or confirm selections.
    if "press" in action.lower():
        if "enter" in action.lower():
            return 0.6
        return 0.2

    # Case C: 'click' action
    # Clicks often result in navigation or significant UI changes.
    if "click" in action.lower():
        # Measure "navigation" by comparing the number of interactive elements (bid tags).
        # A significant change in the number of elements often indicates moving to a new page.
        state_bids = state.count("bid=")
        next_state_bids = next_state.count("bid=")
        
        if state_bids != next_state_bids:
            return 0.5
        
        # If the number of elements is the same, check if the text content has changed,
        # which might suggest a menu was opened or a modal was triggered.
        if next_state.strip() != state.strip():
            return 0.3

    # Default: Actions like noop, scroll, or non-productive clicks return 0.0.
    return 0.0