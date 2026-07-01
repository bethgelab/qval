import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    Q(s, a) is the expected discounted cumulative reward. Since the reward is 1.0 upon 
    goal completion and 0.0 otherwise, the Q-value represents the probability/proximity 
    of reaching the goal from the current transition.
    """
    # Keywords that strongly indicate the task has been completed successfully
    success_keywords = [
        "successfully", "saved", "sent", "added", "confirmed", 
        "completed", "done", "created", "scheduled", "uploaded"
    ]
    # Keywords that indicate a failure or error in the action
    error_keywords = [
        "error", "invalid", "required", "failed", "incorrect", 
        "wrong", "not found", "denied"
    ]
    # Keywords that indicate progress toward a goal (navigating to forms or settings)
    progress_keywords = [
        "form", "input", "edit", "new", "settings", "details", 
        "create", "compose", "add event", "add task", "message"
    ]

    state_lower = state.lower()
    next_state_lower = next_state.lower()

    # 1. Terminal Success Check: 
    # If the next state contains a success keyword that wasn't present in the current state,
    # it's highly likely the goal was just achieved.
    for word in success_keywords:
        if word in next_state_lower and word not in state_lower:
            return 1.0

    # 2. Terminal Failure/Error Check:
    # If the next state introduces an error message, the Q-value for this action is very low.
    for word in error_keywords:
        if word in next_state_lower and word not in state_lower:
            return 0.05

    # 3. Action Analysis:
    # Parse the action to identify the primitive and its arguments.
    # Expected formats: click('bid'), fill('bid', 'text'), press('bid', 'key'), noop(ms), scroll(x, y)
    match = re.search(r"(\w+)\('(.*?)'(?:,\s*'(.*?)')?\)", action)
    if not match:
        # Fallback for actions that don't match the standard pattern (like noop or scroll)
        if "noop" in action or "scroll" in action:
            return 0.1
        return 0.1

    act_type = match.group(1)
    bid = match.group(2)
    val = match.group(3)

    # Base Q-value
    q = 0.1

    if act_type == 'fill':
        # Productive fill: The text entered actually appears in the next state
        if val and val in next_state and val not in state:
            q = 0.5
        else:
            q = 0.2
    elif act_type == 'click':
        # Productive click: The page state changed, indicating navigation or form submission
        if next_state != state:
            q = 0.4
        else:
            # Clicking something that doesn't change the state is usually unproductive
            q = 0.05
    elif act_type == 'press':
        # Pressing a key (like Enter) is often part of a submission process
        q = 0.3
    elif act_type == 'scroll':
        q = 0.1
    elif act_type == 'noop':
        q = 0.05
    else:
        q = 0.1

    # 4. Progress Bonus:
    # Increase Q if the transition led to a state that looks like a "creation" or "editing" page.
    for word in progress_keywords:
        if word in next_state_lower and word not in state_lower:
            q += 0.2

    # Final clamping: Ensure the result is within [0.0, 0.9] because 1.0 is reserved for success.
    return min(max(q, 0.0), 0.9)