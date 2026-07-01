import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the OpenApps environment.
    The Q-value represents the expected return (reaching the goal) given the 
    current state and action, assuming optimal play thereafter.
    """
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()

    # 1. High-confidence Success Detection
    # Look for markers that strongly suggest the task-level goal has been achieved.
    completion_markers = [
        "successfully", "confirmed", "message sent", "event created", 
        "task added", "saved successfully", "completed", "done"
    ]
    if any(marker in next_state_lower for marker in completion_markers):
        return 1.0
    
    # Detect "success" words that appear in next_state but were not in state
    success_keywords = ["success", "saved", "sent", "created", "added", "scheduled"]
    for kw in success_keywords:
        if kw in next_state_lower and kw not in state_lower:
            return 0.95

    # 2. Error Detection
    # If the action leads to a state with clear error indicators, the Q-value drops.
    error_keywords = ["error", "invalid", "required", "failed", "incorrect", "wrong", "not found"]
    if any(kw in next_state_lower for kw in error_keywords):
        # We return a low value but not 0, as it's still possible to recover.
        return 0.1

    # 3. Progress Detection (State Transition Analysis)
    # Check if the action opened a form, revealed a submission button, or moved to a detail page.
    progress_markers = ["input", "text field", "submit", "save", "confirm", "create", "new"]
    # If progress markers appear in next_state that weren't in state, the action was likely productive.
    appeared_markers = [m for m in progress_markers if m in next_state_lower and m not in state_lower]
    if appeared_markers:
        return 0.8

    # 4. Action-Based Heuristics
    # Actions that provide information or commit changes are generally more valuable.
    
    # 'fill' action: Providing data is a prerequisite for almost every OpenApps goal.
    if 'fill' in action_lower:
        return 0.7
    
    # 'press' action: Pressing 'Enter' or similar is often the final step of a sequence.
    if 'press' in action_lower:
        return 0.6
        
    # 'click' action: Value depends on whether it caused a state change.
    if 'click' in action_lower:
        # Significant change in accessibility tree length usually implies a page navigation or modal.
        if abs(len(next_state) - len(state)) > 100:
            return 0.6
        return 0.4
    
    # 'scroll' action: Searching for elements is useful but low-value compared to interacting.
    if 'scroll' in action_lower:
        return 0.3
        
    # 'noop' action: Waiting is generally inefficient unless necessary for loading.
    if 'noop' in action_lower:
        return 0.2

    # Default baseline for ambiguous transitions.
    return 0.5