import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an OpenApps environment.
    The estimate is based on detecting goal achievement, progress in forms, and 
    navigation efficiency.
    """
    # 1. Success Check (Immediate Goal Achievement)
    # If the action leads to a state where the task is confirmed as complete.
    # We look for specific success phrases common in web applications.
    success_indicators = [
        "message sent", "event created", "task added", "todo added",
        "successfully", "confirmed", "saved successfully", 
        "sent successfully", "added successfully", "task completed"
    ]
    
    next_lower = next_state.lower()
    for indicator in success_indicators:
        if indicator in next_lower:
            return 1.0

    score = 0.0
    action_lower = action.lower()

    # 2. Progress Heuristics
    # A. Form Filling Progress
    # Detect pattern: fill('bid', 'text')
    fill_match = re.search(r"fill\(['\"](\d+)['\"],\s*['\"](.*?)['\"]\)", action_lower)
    if fill_match:
        bid, text = fill_match.groups()
        # If the text provided in the fill action is now visible in the next state, 
        # it is a very strong indicator of progress.
        if text and text.lower() in next_lower:
            score += 0.6
        else:
            score += 0.3
    
    # B. Clicking Progress
    # Detect pattern: click('bid')
    elif "click" in action_lower:
        # If the state changes significantly, it likely means navigation or a page update.
        if next_state.strip() != state.strip():
            score += 0.4
        
        # Attempt to identify if the clicked element is a "finishing" action (e.g., 'Submit').
        # We extract the bid from the action and look at the surrounding context in the state.
        bid_match = re.search(r"['\"](\d+)['\"]", action_lower)
        if bid_match:
            bid = bid_match.group(1)
            idx = state.find(bid)
            if idx != -1:
                # Check a window around the bid in the accessibility tree for finishing keywords.
                context = state[max(0, idx-40):min(len(state), idx+40)].lower()
                finishing_words = ['submit', 'send', 'save', 'create', 'add', 'confirm', 'done', 'ok']
                if any(word in context for word in finishing_words):
                    score += 0.4

    # C. Key Press Progress (e.g., pressing Enter to submit)
    elif "press" in action_lower:
        if next_state.strip() != state.strip():
            score += 0.3

    # 3. Penalties for Ineffective Actions
    # If the action was a no-op, it has no value.
    if "noop" in action_lower:
        score = 0.0
    # If the state did not change, the action was likely ineffective or redundant.
    elif next_state.strip() == state.strip():
        score -= 0.3

    # 4. Final Value Clamping
    # Returns a value in [0.0, 0.99]. 1.0 is strictly reserved for confirmed success.
    return max(0.0, min(0.99, score))