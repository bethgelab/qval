def signal_function(state: str, action: str, next_state: str) -> float:
    # Normalize text for case-insensitive matching
    s_text = state.lower()
    a_text = action.lower()
    ns_text = next_state.lower()

    # 1. Check for Success (Terminal State)
    # If the next state indicates the task is done, Q = 1.0
    success_indicators = [
        "task complete", "success", "goal achieved", "done", 
        "congratulations", "correct"
    ]
    if any(indicator in ns_text for indicator in success_indicators):
        return 1.0

    # 2. Check for Failure/Error
    # If the environment returns an error message, the action was invalid.
    error_indicators = [
        "can't", "cannot", "invalid", "wrong", "error", 
        "not allowed", "cannot be done", "doesn't"
    ]
    if any(indicator in ns_text for indicator in error_indicators):
        return 0.0

    # 3. Check for No-Op (State unchanged)
    # If state and next_state are identical, no progress was made.
    if state == next_state:
        return 0.05

    # 4. Estimate Action Quality based on Type
    # Manipulation is generally higher value as it progresses towards goal
    nav_verbs = ["go to", "walk", "move to", "enter"]
    manip_verbs = ["take", "put", "move", "clean", "wipe", "open", "close", "break", "eat", "drink"]
    obs_verbs = ["look", "examine", "read", "check"]

    base_score = 0.0

    if any(verb in a_text for verb in nav_verbs):
        base_score = 0.2
    elif any(verb in a_text for verb in manip_verbs):
        base_score = 0.6
    elif any(verb in a_text for verb in obs_verbs):
        base_score = 0.1
    else:
        # Unknown or invalid action
        base_score = 0.05

    # 5. Object Consistency Bonus
    # If the action mentions specific objects found in the state description,
    # it implies the agent knows the environment, slightly increasing value.
    common_objects = [
        "mug", "pot", "pan", "bowl", "cup", "plate", "bread", 
        "tomato", "lettuce", "knife", "spoon", "fork", "detergent", 
        "soap", "spray", "sink", "countertop", "microwave", "fridge", 
        "stove", "oven", "table", "desk", "drawer", "trashcan"
    ]
    
    action_words = a_text.split()
    state_words = s_text.split()
    
    action_has_object = any(word in common_objects for word in action_words)
    state_has_object = any(word in common_objects for word in state_words)

    if action_has_object and state_has_object:
        base_score += 0.1

    # Clamp result to [0, 1]
    return min(1.0, max(0.0, base_score))