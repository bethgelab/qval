import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The estimate is based on identifying progress towards a goal (like picking up 
    objects, navigating, or cleaning) and terminal states.
    """
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()

    # 1. Check for terminal success/failure
    # If the next state indicates the task is completed.
    if any(x in ns for x in ["success", "task completed", "goal reached", "you win", "you have successfully"]):
        return 1.0
    
    # If the next state indicates the episode has failed.
    if any(x in ns for x in ["failed", "the episode ends", "you lose"]):
        return 0.0

    # 2. Utility for comparing state changes
    def clean_text(text):
        return re.sub(r'[^a-z0-9]', '', text)

    # If the state remains effectively unchanged, the action was likely useless.
    if clean_text(s) == clean_text(ns):
        return 0.0

    # 3. Progress estimation based on action type and its result in next_state
    progress_value = 0.0
    
    # Helper to strip 'the' and whitespace from target objects/locations
    def extract_target(target_str):
        return re.sub(r'\bthe\b', '', target_str).strip()

    # Case: Navigation (e.g., "go to kitchen")
    if "go to" in a:
        target = extract_target(a.split("go to")[-1])
        if target and target in ns:
            progress_value = 0.4
    
    # Case: Picking up/Taking (e.g., "pick up apple", "take apple")
    elif "pick up" in a or "take" in a:
        if "pick up" in a:
            obj = extract_target(a.split("pick up")[-1])
        else:
            obj = extract_target(a.split("take")[-1])
        
        if obj and (("holding" in ns or "have" in ns) and obj in ns):
            progress_value = 0.6
    
    # Case: Putting (e.g., "put apple in bowl")
    elif "put" in a:
        if " in " in a:
            parts = a.split("put")[-1].split(" in ")
            if len(parts) == 2:
                obj = extract_target(parts[0])
                loc = extract_target(parts[1])
                if obj and loc and obj in ns and loc in ns:
                    progress_value = 0.8
    
    # Case: Cleaning (e.g., "clean apple")
    elif "clean" in a:
        obj = extract_target(a.split("clean")[-1])
        if obj and obj in ns and "clean" in ns:
            progress_value = 0.7

    # Case: Opening/Closing (e.g., "open fridge")
    elif "open" in a:
        obj = extract_target(a.split("open")[-1])
        if obj and obj in ns and ("open" in ns or "is open" in ns):
            progress_value = 0.5
    elif "close" in a:
        obj = extract_target(a.split("close")[-1])
        if obj and obj in ns and ("closed" in ns or "is closed" in ns):
            progress_value = 0.5
            
    # Case: Dropping (e.g., "drop apple")
    elif "drop" in a:
        obj = extract_target(a.split("drop")[-1])
        if obj and obj in ns:
            progress_value = 0.5

    # 4. Final evaluation of the progress
    if progress_value > 0:
        return progress_value

    # If no progress was detected, check if the action explicitly failed.
    # If the environment returns an error like "You cannot pick up...", return 0.0.
    if any(x in ns for x in ["cannot", "don't", "does not", "not possible", "not found", "not able"]):
        return 0.0
        
    # If no progress and no explicit error, return a low value for neutral/exploratory actions.
    return 0.1