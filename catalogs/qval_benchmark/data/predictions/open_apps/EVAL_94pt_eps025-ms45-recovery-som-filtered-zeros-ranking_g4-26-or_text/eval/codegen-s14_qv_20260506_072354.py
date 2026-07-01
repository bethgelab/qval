import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state-action-next_state transition in the OpenApps environment.
    The estimate is based on detecting goal achievement, error states, progress via input/interaction,
    and state transitions.
    """
    # 1. Immediate Success Check
    # If the next_state contains words indicating completion or the manifestation of a goal.
    success_indicators = [
        'success', 'completed', 'done', 'saved', 'sent', 'added', 'created', 
        'message sent', 'event created', 'todo added', 'task completed'
    ]
    if any(indicator in next_state.lower() for indicator in success_indicators):
        return 1.0

    # 2. Error/Failure Check
    # If the next_state contains words indicating something went wrong.
    error_indicators = [
        'error', 'failed', 'invalid', 'incorrect', 'not found', 
        'required', 'must be', 'cannot', 'warning'
    ]
    if any(indicator in next_state.lower() for indicator in error_indicators):
        return 0.0

    # 3. Parse Action Details
    # Action format is typically: action_type(bid=..., text='...')
    action_type = ""
    action_text = ""
    
    type_match = re.match(r"^(\w+)\(", action)
    if type_match:
        action_type = type_match.group(1)
        
    text_match = re.search(r"text=['\"](.*?)['\"]", action)
    if text_match:
        action_text = text_match.group(1)

    # 4. Evaluate Progress Based on Action and State Change
    
    # If noop or nothing happened
    if action_type == "noop" or next_state.strip() == state.strip():
        return 0.1

    # Case: Filling information
    if action_type == "fill":
        # If the text being filled appears in the next state, it's a strong sign of progress
        if action_text and action_text in next_state:
            return 0.8
        # If it's a fill action but the state changed (e.g., field updated), it's likely progress
        if next_state != state:
            return 0.4
        return 0.2

    # Case: Clicking or Pressing (Submitting/Navigating)
    if action_type in ["click", "press"]:
        # If the length of the state string changes significantly, it implies a page change or modal open
        # We use a relative difference threshold
        len_diff = abs(len(next_state) - len(state))
        if len_diff > (len(state) * 0.1):
            return 0.6
        # If content changed significantly (heuristic via word count or similar)
        # but length is similar, check if any words are new.
        # Since we can't use set() easily on large text without overhead, 
        # we check if the next state contains new content.
        if next_state != state:
            return 0.4
        return 0.2

    # Case: Scrolling
    if action_type == "scroll":
        # Scrolling is progress if it reveals new content
        if next_state != state:
            return 0.3
        return 0.1

    # Default fallback for unknown actions or small changes
    if next_state != state:
        return 0.3
    
    return 0.1