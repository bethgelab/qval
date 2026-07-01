import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment state.
    The value represents the expected discounted reward (probability of goal completion).
    """
    state_lower = state.lower()
    
    # 1. Goal Achievement Markers (Highest Value)
    # Look for clear indicators that the task has been completed.
    # We prioritize phrases that suggest a completed action rather than a command.
    completion_patterns = [
        r"successfully", 
        r"has been (created|sent|saved|added)",
        r"was (created|sent|saved|added)",
        r"completion",
        r"task complete",
        r"message sent",
        r"event saved",
        r"goal achieved"
    ]
    for pattern in completion_patterns:
        if re.search(pattern, state_lower):
            return 1.0

    # 2. Final Action Step (High Value)
    # Markers of being on the final interaction (e.g., a 'Save' or 'Send' button)
    # and having already provided input (existence of textboxes or filled values).
    final_button_patterns = [
        r"button.*?(save|submit|send|confirm|create|add|done)",
        r"link.*?(save|submit|send|confirm|create|add|done)"
    ]
    
    has_final_button = False
    for pattern in final_button_patterns:
        if re.search(pattern, state_lower):
            has_final_button = True
            break
            
    if has_final_button:
        # If we see a final button and signs of data entry, we are very close.
        if any(x in state_lower for x in ["input", "textbox", "value=", "text="]):
            return 0.8
        # If we see the button but maybe no data yet, we are closer than the start.
        return 0.6

    # 3. Initial Navigation/Entry Step (Medium Value)
    # Markers of starting the task flow (e.g., 'New Event', 'Compose Message').
    entry_button_patterns = [
        r"button.*?(new|compose|add|create|plus)",
        r"link.*?(new|compose|add|create|plus)"
    ]
    for pattern in entry_button_patterns:
        if re.search(pattern, state_lower):
            return 0.4
    
    # 4. Error/Stuck States (Low Value)
    # Markers indicating the agent is in a failure state or missing information.
    error_patterns = [
        r"error", 
        r"invalid", 
        r"required", 
        r"failed", 
        r"incorrect",
        r"not found",
        r"missing"
    ]
    for pattern in error_patterns:
        if re.search(pattern, state_lower):
            return 0.1

    # 5. Default/Baseline (Lowest Value)
    # The state is neither clearly successful nor clearly progressing.
    return 0.0