import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a given state transition in the OpenApps environment.
    The value reflects the estimated expected discounted return, where 1.0 is the 
    maximum reward for goal achievement.
    """
    
    # 1. Goal Completion Heuristics
    # These keywords strongly suggest the task has been successfully completed.
    success_keywords = [
        "successfully", "created", "sent", "added", "completed", 
        "saved", "confirmed", "done", "success", "finished"
    ]
    
    # We check if these keywords appear in the next_state but were not present in the state,
    # signaling a transition to a success confirmation screen.
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    
    for word in success_keywords:
        if word in next_state_lower and word not in state_lower:
            return 0.98

    # 2. Failure/Error Heuristics
    # These keywords suggest a mistake was made or the action was invalid.
    error_keywords = [
        "error", "invalid", "required", "failed", "incorrect", 
        "wrong", "cannot", "unable to", "missing"
    ]
    for word in error_keywords:
        if word in next_state_lower and word not in state_lower:
            return 0.02

    # 3. Action-based and State-based Value Estimation
    # Start with a baseline Q-value for any valid action.
    q_val = 0.2

    # Extract the 'bid' from action primitives like click('12'), fill('12', 'text'), etc.
    # The action format is usually function('bid', ...)
    bid_match = re.search(r"'(.*?)'", action)
    if bid_match:
        bid = bid_match.group(1)
        # Attempt to locate the element associated with the bid in the state accessibility tree.
        # We look for patterns like [12] followed by the element's label.
        element_pattern = rf"\[{bid}\]\s*(.*?)(?=\n|\[|$)"
        element_match = re.search(element_pattern, state)
        
        if element_match:
            element_text = element_match.group(1).lower()
            
            # Productive interaction keywords
            positive_keywords = ["save", "submit", "create", "send", "add", "ok", "confirm", "next", "yes"]
            negative_keywords = ["cancel", "delete", "back", "remove", "discard", "no", "clear"]
            
            if any(w in element_text for w in positive_keywords):
                q_val += 0.3
            elif any(w in element_text for w in negative_keywords):
                q_val -= 0.2

    # Heuristics based on the type of BrowserGym primitive used
    if action.startswith("fill"):
        # Filling forms is usually a necessary step toward the goal.
        q_val += 0.2
    elif action.startswith("press"):
        # Pressing 'Enter' is often a submission.
        if "'Enter'" in action:
            q_val += 0.2
    elif action.startswith("noop"):
        # No-ops are generally inefficient unless waiting for a load.
        q_val -= 0.1

    # 4. State Transition Logic
    # A change in state (that isn't an error) usually indicates progress.
    if len(next_state) != len(state):
        q_val += 0.1
        
    # 5. Bound the resulting Q-value between 0.0 and 1.0
    # We cap it slightly below 1.0 unless a success keyword was found to avoid overestimation.
    return max(0.0, min(q_val, 0.85))