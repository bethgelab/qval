import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value by analyzing the semantic change in the accessibility tree
    and identifying progress toward task completion.
    """
    # 1. Check for Terminal States (Success/Error)
    # The reward is 1.0 upon goal achievement. We look for semantic indicators in the next_state.
    success_patterns = [r"success", r"done", r"sent", r"created", r"added", r"saved", r"completed", r"confirmed"]
    error_patterns = [r"error", r"failed", r"invalid", r"wrong", r"not found", r"alert", r"failure"]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 1.0
    for pattern in error_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            return 0.0

    # 2. Detect Ineffective Actions
    # If the state hasn't changed or the action is a no-op, the expected reward is 0.
    if action.startswith("noop") or next_state == state:
        return 0.0

    # 3. Extract Action Type
    # BrowserGym actions are formatted as function_name(params)
    action_type = action.split("(")[0]

    # 4. Feature Extraction
    # We represent the state as a set of (attribute, value) pairs to detect semantic changes.
    def get_features(s: str):
        # Matches patterns like role="button" or name='Submit'
        return set(re.findall(r'(\w+)=["\']([^"\']*)["\']', s))

    state_features = get_features(state)
    next_features = get_features(next_state)

    # 5. Heuristic Scoring based on Action Type and State Transition
    score = 0.1  # Default score for an action that caused some change

    if action_type == "fill":
        # A 'fill' action is productive if it results in a change to the 'value' attribute
        state_vals = {v for k, v in state_features if k == 'value'}
        next_vals = {v for k, v in next_features if k == 'value'}
        if next_vals != state_vals:
            score = 0.5
        else:
            score = 0.05

    elif action_type == "click":
        # A 'click' is productive if it changes element states (e.g., checked, expanded, pressed)
        # or alters the set of visible interactive elements (structural change).
        if next_features != state_features:
            # If many attributes changed, it's likely a significant navigation or UI update
            score = 0.6
        else:
            score = 0.1

    elif action_type == "press":
        # 'press' (e.g., Enter key) often submits forms or triggers structural changes
        if next_features != state_features:
            score = 0.5
        else:
            score = 0.2

    elif action_type == "scroll":
        # 'scroll' is productive if it reveals new elements (increased 'bid' count)
        state_bids = len(re.findall(r'bid=', state))
        next_bids = len(re.findall(r'bid=', next_state))
        if next_bids > state_bids:
            score = 0.3
        else:
            score = 0.05

    # 6. Progress Boost
    # If the action causes a large delta in the accessibility tree's features, 
    # it implies significant progress (like navigating to a new page).
    if len(next_features - state_features) > 5:
        score = min(score + 0.2, 0.8)

    # Ensure the result is within [0.0, 1.0]
    return float(min(max(score, 0.0), 1.0))