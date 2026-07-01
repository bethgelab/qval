import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for an action taken in an ALFWorld environment.
    The Q-value is an approximation of the expected discounted reward.
    """
    s_l = state.lower()
    a_l = action.lower()
    ns_l = next_state.lower()

    # 1. Immediate Success/Terminal State Check
    # If the next state indicates the task is complete, the Q-value is 1.0.
    success_keywords = ["success", "task complete", "goal met", "you have finished", "finished the task"]
    if any(kw in ns_l for kw in success_keywords):
        return 1.0

    # 2. Check for Failed Action
    # If the action resulted in a failure message, the Q-value is low.
    failure_keywords = [
        "you can't", "nothing happens", "is not here", "is not reachable",
        "not available", "i don't understand", "is not in", "is not on"
    ]
    if any(kw in ns_l for kw in failure_keywords):
        return 0.0

    # 3. Base score representing progress
    score = 0.1

    # 4. Action-Outcome Heuristics
    # A. Movement: "go to <room>"
    if "go to" in a_l:
        # Extract room: "go to the kitchen" -> "kitchen"
        room = a_l.split("go to")[-1].replace("the ", "").strip()
        if room in ns_l:
            score += 0.3

    # B. Pickup: "pickup <obj>" or "take <obj>"
    if "pickup" in a_l or "take" in a_l:
        # Extract object: "pickup the apple" -> "apple"
        obj = a_l.split("pickup")[-1].split("take")[-1].replace("the ", "").strip()
        # If the agent is now holding the object
        if "holding" in ns_l and obj in ns_l:
            score += 0.5
        elif "holding" in ns_l and any(word in ns_l for word in obj.split() if len(word) > 2):
            # Partial match fallback
            score += 0.4

    # C. Put: "put <obj> in <loc>"
    if "put" in a_l and "in" in a_l:
        try:
            # Split "put the apple in the fridge" into "apple" and "fridge"
            parts = a_l.split("put")[-1].split("in")
            if len(parts) == 2:
                obj = parts[0].replace("the ", "").strip()
                loc = parts[1].replace("the ", "").strip()
                if obj in ns_l and loc in ns_l:
                    score += 0.7  # High value as this is often the final step
                elif loc in ns_l:
                    score += 0.3
        except IndexError:
            pass

    # D. Drop: "drop <obj>"
    if "drop" in a_l:
        obj = a_l.split("drop")[-1].replace("the ", "").strip()
        # If object is no longer being held, it's likely part of a transition
        if "holding" not in ns_l or obj not in ns_l:
            score += 0.3

    # E. Cleaning / Opening / Closing
    if "clean" in a_l:
        obj = a_l.split("clean")[-1].replace("the ", "").strip()
        if "clean" in ns_l and obj in ns_l:
            score += 0.5
    if "open" in a_l:
        obj = a_l.split("open")[-1].replace("the ", "").strip()
        if "open" in ns_l and obj in ns_l:
            score += 0.4
    if "close" in a_l:
        obj = a_l.split("close")[-1].replace("the ", "").strip()
        if "closed" in ns_l and obj in ns_l:
            score += 0.4

    # 5. General State Change Progress
    # If the agent started holding something in this step
    if "holding" in ns_l and "holding" not in s_l:
        score += 0.2

    # Cap the final Q-value at 1.0
    return min(score, 1.0)