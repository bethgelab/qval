import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    The Q-value represents the expected discounted cumulative reward, which in this binary 
    reward environment is effectively a measure of progress toward the task goal.
    """
    # 1. Check for immediate success indicators in the next state.
    # Since reward is 1.0 only on goal achievement, high values here indicate a likely completion.
    success_keywords = ["successfully", "saved", "created", "sent", "confirmed", "done", "completed", "success", "deleted", "updated"]
    next_state_lower = next_state.lower()
    if any(kw in next_state_lower for kw in success_keywords):
        return 1.0

    # 2. Check for failure indicators in the next state.
    # Error messages suggest the action was incorrect or the state is unfavorable.
    error_keywords = ["error", "invalid", "required", "failed", "incorrect", "cannot"]
    if any(kw in next_state_lower for kw in error_keywords):
        return 0.0

    # 3. Penalize idempotent actions.
    # If the state hasn't changed after an action, it is likely a redundant or failed interaction.
    if state == next_state:
        return 0.0

    # 4. Analyze the action taken to estimate progress.
    # Action patterns: click('bid'), fill('bid', 'text'), press('bid', 'key'), noop(ms), scroll(x, y)
    
    # 'fill' actions typically represent progress in completing a required form.
    if action.startswith("fill"):
        fill_match = re.search(r"fill\(\s*['\"](\d+)['\"]\s*,\s*['\"](.*?)['\"]\s*\)", action)
        if fill_match:
            text_filled = fill_match.group(2)
            # If the filled text now appears in the state, the action was likely successful.
            if text_filled and text_filled in next_state:
                return 0.6
        return 0.4

    # 'press' actions, specifically Enter, often act as form submissions.
    if action.startswith("press"):
        if "Enter" in action:
            return 0.7
        return 0.3

    # 'click' actions are high value if they target "submit-style" buttons.
    if action.startswith("click"):
        click_match = re.search(r"click\(\s*['\"](\d+)['\"]\s*\)", action)
        if click_match:
            bid = click_match.group(1)
            # Extract the element text associated with the bid from the accessibility tree.
            # Format assumed: [bid] ElementType "Label"
            pattern = rf"\[{bid}\]\s*(.*?)(?=\s*\[\d+\]|$)"
            element_match = re.search(pattern, state, re.DOTALL)
            if element_match:
                element_text = element_match.group(1).lower()
                submit_keywords = ["save", "submit", "add", "send", "create", "ok", "confirm", "done", "update", "delete"]
                nav_keywords = ["new", "create", "add", "plus", "compose", "edit", "settings"]
                
                if any(kw in element_text for kw in submit_keywords):
                    return 0.8
                if any(kw in element_text for kw in nav_keywords):
                    return 0.4
        return 0.2

    # 'noop' and 'scroll' are low-value utility actions.
    if action.startswith("noop") or action.startswith("scroll"):
        return 0.1

    # Default baseline value for any other action.
    return 0.1