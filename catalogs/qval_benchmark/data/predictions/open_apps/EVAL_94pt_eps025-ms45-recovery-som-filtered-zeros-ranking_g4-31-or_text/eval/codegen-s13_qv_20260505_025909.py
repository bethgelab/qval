import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the OpenApps environment.
    Q-value represents the expected discounted cumulative reward.
    Since reward is binary (1.0 on goal achievement), this function estimates
    the likelihood and efficiency of reaching the goal from the current state
    given the action.
    """
    # 1. Immediate Success Detection
    # These keywords typically indicate the task-level goal has been achieved.
    success_keywords = ["successfully", "created", "added", "sent", "saved", "completed", "confirmed", "welcome", "success", "done"]
    if any(kw in next_state.lower() for kw in success_keywords):
        # Highest value as the reward 1.0 is likely achieved in the next check.
        return 0.98

    # 2. Immediate Failure/Error Detection
    # Error messages suggest the action was counter-productive or the state is unfavorable.
    error_keywords = ["error", "failed", "invalid", "incorrect", "required", "wrong", "not found", "unable to"]
    if any(kw in next_state.lower() for kw in error_keywords):
        return 0.1

    # 3. Stagnation Detection
    # If the state hasn't changed, the action was likely a no-op or invalid.
    if state == next_state:
        return 0.05

    # 4. Informed Action Analysis
    # Analyze the specific element being interacted with using the bid.
    # BrowserGym actions typically format as action('bid', ...) or action('bid').
    bid_match = re.search(r"'(.*?)'", action)
    if bid_match:
        bid = bid_match.group(1)
        # Search for the element with the corresponding bid in the accessibility tree representation.
        # Expected format: [bid] ElementType "Text"
        element_pattern = rf"\[{re.escape(bid)}\]\s*([^\]\n]*)"
        element_match = re.search(element_pattern, state)
        if element_match:
            element_text = element_match.group(1).lower()
            # Interaction with 'Submit', 'Save', 'Add' buttons is usually high-value progress.
            high_value_keywords = ["save", "submit", "add", "send", "create", "ok", "confirm", "yes", "login", "sign in", "enter", "finish"]
            if any(kw in element_text for kw in high_value_keywords):
                # Higher value for purposeful progress toward completion.
                return 0.85

    # 5. General Action Heuristics
    # Different action types have different typical contributions to goal progression.
    if "click" in action:
        # Clicking generally moves the agent to a new page or triggers a state change.
        return 0.6
    if "fill" in action:
        # Filling forms is a necessary prerequisite for most goals in OpenApps.
        return 0.5
    if "press" in action:
        # Key presses (like Enter) are often useful but slightly less predictable than clicks.
        return 0.4
    if "scroll" in action or "noop" in action:
        # Scrolling and no-ops are low-value maintenance actions.
        return 0.2

    # Default baseline estimate for unclassified actions.
    return 0.3