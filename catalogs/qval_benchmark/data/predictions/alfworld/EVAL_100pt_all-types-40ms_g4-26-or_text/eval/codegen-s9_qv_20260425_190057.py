import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for taking an action in a given state.
    The Q-value is an approximation based on the potential for progress toward the goal.
    """
    s_low = state.lower()
    a_low = action.lower()
    ns_low = next_state.lower()

    # 1. Check for terminal success in the next state
    # ALFWorld usually indicates success with specific phrases.
    success_indicators = [
        "task is complete",
        "goal reached",
        "successfully",
        "you have achieved",
        "your goal is met"
    ]
    if any(ind in ns_low for ind in success_indicators):
        return 1.0

    # 2. Check for invalid actions or failed attempts
    # If the action failed, the environment often provides immediate feedback.
    invalid_indicators = [
        "you cannot",
        "is not here",
        "is not in",
        "don't see",
        "don't have",
        "you are not holding",
        "is not a",
        "you don't"
    ]
    if any(ind in ns_low for ind in invalid_indicators):
        return 0.0
    
    # If the action resulted in no state change, it's generally not a productive step.
    if ns_low == s_low:
        return 0.0

    # 3. Scoring Progress
    # We start with a baseline score for any valid, state-changing action.
    score = 0.1

    # Pattern: "take [object]" or "pick up [object]"
    if "take" in a_low or "pick" in a_low or "grab" in a_low:
        if "holding" in ns_low or "you are holding" in ns_low:
            # Check if significant words from the action (excluding verbs) are in the next state.
            action_words = [w for w in a_low.split() if len(w) > 3 and w not in ["take", "pick", "grab"]]
            if any(w in ns_low for w in action_words):
                score += 0.6

    # Pattern: "go to [location]" or "move to [location]"
    if "go" in a_low or "move" in a_low or "walk" in a_low:
        if "you are in" in ns_low or "you are now in" in ns_low:
            action_words = [w for w in a_low.split() if len(w) > 3 and w not in ["go", "to", "move", "walk"]]
            if any(w in ns_low for w in action_words):
                score += 0.4

    # Pattern: "put [object] in [location]"
    if "put" in a_low and " in " in a_low:
        if " in " in ns_low:
            # We split the action to identify the intended object and container.
            parts = a_low.split(" in ")
            obj_part = parts[0]
            loc_part = parts[1]
            
            obj_words = [w for w in obj_part.split() if len(w) > 2 and w not in ["put", "the", "a", "an"]]
            loc_words = [w for w in loc_part.split() if len(w) > 2 and w not in ["the", "a", "an"]]
            
            # If both the object and location are represented in the new state's description of placement.
            if (any(ow in ns_low for ow in obj_words) and any(lw in ns_low for lw in loc_words)):
                score += 0.7

    # Pattern: "clean [object]"
    if "clean" in a_low:
        if "clean" in ns_low or "is now clean" in ns_low or "is clean" in ns_low:
            action_words = [w for w in a_low.split() if len(w) > 3 and w != "clean"]
            if any(w in ns_low for w in action_words):
                score += 0.6

    # Pattern: "open [object]"
    if "open" in a_low:
        if "open" in ns_low or "is open" in ns_low:
            action_words = [w for w in a_low.split() if len(w) > 3 and w != "open"]
            if any(w in ns_low for w in action_words):
                score += 0.4

    # Pattern: "drop [object]"
    if "drop" in a_low:
        if "is on" in ns_low or "is in" in ns_low or "is a" in ns_low:
            action_words = [w for w in a_low.split() if len(w) > 3 and w != "drop"]
            if any(w in ns_low for w in action_words):
                score += 0.3

    # Cap the score to ensure it stays within [0, 1]. 
    # We use 0.95 as the max for non-terminal success to allow for terminal 1.0.
    return min(score, 0.95)