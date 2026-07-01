import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    The Q-value is an approximation of the expected cumulative reward (reaching the goal).
    """
    # 1. Parse the action type and any text arguments
    action_type = action.split('(')[0].strip() if '(' in action else action.strip()
    
    # Extract text if the action is 'fill'
    filled_text = None
    text_match = re.search(r"text=['\"](.*?)['\"]", action)
    if text_match:
        filled_text = text_match.group(1)

    # Normalize strings for easier comparison
    next_state_lower = next_state.lower()
    state_lower = state.lower()

    # 2. Immediate Goal/Error detection (High Impact)
    # We look for success keywords that appear in the next state but were not in the previous state.
    success_keywords = [
        "success", "task added", "event created", "message sent", 
        "item added", "directions found", "saved successfully", 
        "completed task", "confirmed", "sent successfully"
    ]
    for kw in success_keywords:
        if kw in next_state_lower and kw not in state_lower:
            return 1.0
            
    # Check for error/failure indicators newly introduced
    error_keywords = ["error", "invalid", "failed", "not found", "try again", "incorrect"]
    for kw in error_keywords:
        if kw in next_state_lower and kw not in state_lower:
            return 0.0

    # 3. Productivity Heuristics (Medium Impact)
    
    # If 'fill' was used and the text is now visible in the UI, it's highly productive.
    if action_type == "fill" and filled_text and len(filled_text) > 1:
        if filled_text.lower() in next_state_lower:
            return 0.8
            
    # Check if the action caused any change in the accessibility tree
    # A change in the state text indicates navigation, element updates, or input.
    if next_state.strip() != state.strip():
        # If the change was caused by a click, press, or scroll, it's likely progress.
        if action_type in ["click", "press", "scroll"]:
            return 0.6
        # A change caused by 'fill' is good, even if the text doesn't immediately show up.
        elif action_type == "fill":
            return 0.5
        else:
            return 0.4
    else:
        # The state remained unchanged.
        if action_type == "noop":
            # No-op is a valid but non-progressive action.
            return 0.1
        else:
            # An action that results in no change is usually an invalid click or a failed interaction.
            return 0.0

    # Default fallback value
    return 0.2