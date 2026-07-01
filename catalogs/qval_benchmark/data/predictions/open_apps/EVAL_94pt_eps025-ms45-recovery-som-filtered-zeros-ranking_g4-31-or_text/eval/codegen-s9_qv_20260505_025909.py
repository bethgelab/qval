import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for an OpenApps environment based on 
    state transitions and action intent.
    """
    # 1. Detect Terminal Success
    # If the next state contains confirmation keywords that weren't in the previous state,
    # it's highly likely the goal was achieved.
    success_keywords = ['successfully', 'saved', 'created', 'sent', 'completed', 'confirmed', 'done', 'added']
    for kw in success_keywords:
        if kw in next_state.lower() and kw not in state.lower():
            return 1.0

    # 2. Detect Failure/Errors
    # If the action led to a new error message, the Q-value is very low.
    error_keywords = ['error', 'invalid', 'required', 'failed', 'unable to']
    for kw in error_keywords:
        if kw in next_state.lower() and kw not in state.lower():
            return 0.0

    # 3. Base Q-value and Action-Based Progress
    # We start with a low baseline and add value based on signs of progress.
    q_est = 0.2
    
    # Normalize action to identify type and target bid
    # Expected formats: click('1'), fill('2', 'text'), press('3', 'Enter'), noop(100), scroll(0, 10)
    action_type = None
    bid = None
    
    if action.startswith("click"):
        action_type = "click"
        match = re.search(r"click\(['\"]([^'\"]+)['\"]\)", action)
        if match:
            bid = match.group(1)
    elif action.startswith("fill"):
        action_type = "fill"
        match = re.search(r"fill\(['\"]([^'\"]+)['\"]", action)
        if match:
            bid = match.group(1)
    elif action.startswith("press"):
        action_type = "press"
        match = re.search(r"press\(['\"]([^'\"]+)['\"]", action)
        if match:
            bid = match.group(1)
    elif action.startswith("noop"):
        action_type = "noop"
    elif action.startswith("scroll"):
        action_type = "scroll"

    # Heuristic weights for different action outcomes
    if action_type == "fill":
        # Filling a form field is usually a strong indicator of progress toward a goal
        q_est += 0.3
    elif action_type == "click" and bid:
        # Check the accessibility tree (state) for the element that was clicked
        # Search for bid association with key verbs like 'Save', 'Submit', 'Add'
        submit_pattern = rf"(?:bid\s*[:=]\s*{re.escape(bid)}|\[bid={re.escape(bid)}\]).*?(Save|Submit|Add|Send|Create|Done|Confirm)"
        if re.search(submit_pattern, state, re.IGNORECASE):
            q_est += 0.5  # High value for submission actions
        else:
            # Check if it was a navigation action to enter a creation flow
            nav_pattern = rf"(?:bid\s*[:=]\s*{re.escape(bid)}|\[bid={re.escape(bid)}\]).*?(New|Create|Add|Plus|Compose)"
            if re.search(nav_pattern, state, re.IGNORECASE):
                q_est += 0.3  # Moderate value for entering a form
            elif next_state != state:
                q_est += 0.1  # Small value for generic state change
    elif action_type == "press":
        if "Enter" in action:
            q_est += 0.4  # Enter key often acts as a submission
    elif action_type == "noop":
        q_est -= 0.1

    # 4. State-based Progress Analysis
    # If we transition from a list view to a form view, that's positive progress
    if "textbox" in next_state.lower() and "textbox" not in state.lower():
        q_est += 0.2
    
    # Penalize actions that result in no change to the environment
    if state == next_state:
        q_est -= 0.2

    # Final clamping to [0.0, 1.0]
    return max(0.0, min(1.0, q_est))