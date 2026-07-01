import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for an ALFWorld state-action pair by analyzing the 
    transition from the current state to the next state via text-based analysis.
    """
    ns = next_state.lower()
    s = state.lower()
    a = action.lower()

    # 1. Immediate Success Detection
    # If the text indicates the task has been completed.
    success_keywords = ["success", "task completed", "goal reached"]
    if any(keyword in ns for keyword in success_keywords):
        return 1.0
    
    # 2. Immediate Failure Detection
    # If the action was invalid, the object was unavailable, or nothing changed.
    fail_patterns = [
        "you cannot", 
        "nothing happens", 
        "is not reachable", 
        "is not here", 
        "you are not holding", 
        "you are not in"
    ]
    if any(pattern in ns for pattern in fail_patterns):
        return 0.0
    
    # 3. Productive Action Detection (Progress-based Heuristics)
    
    # A. Picking up/Taking an object
    # Expected progression: "pick up the apple" -> "you are holding the apple"
    if "pick up" in a or "take" in a:
        # Extract the target object from the action command
        obj = a.split("pick up")[-1].split("take")[-1].replace("the", "").strip()
        if obj and obj in ns:
            # Confirm the object is actually held in the next state
            if "holding" in ns or "you have" in ns or "you picked up" in ns:
                return 0.7
    
    # B. Moving/Navigating to a location
    # Expected progression: "go to the kitchen" -> "you are in the kitchen"
    if "go to" in a or "move to" in a:
        # Extract the target room from the action command
        room = a.split("go to")[-1].split("move to")[-1].replace("the", "").strip()
        if room and room in ns:
            # Confirm the agent's location has changed to the target room
            if "you are in" in ns or "you are now in" in ns:
                return 0.4

    # C. Putting/Placing an object in a location
    # Expected progression: "put the apple in the fridge" -> "the apple is in the fridge"
    if "put" in a or "place" in a:
        if " in " in a:
            parts = a.split(" in ")
            # Extract object and target location
            # Example: "put apple in fridge" -> obj="apple", target="fridge"
            obj_part = parts[0].replace("put", "").replace("place", "").replace("the", "").strip()
            target_part = parts[1].replace("the", "").strip()
            if obj_part in ns and target_part in ns:
                # High probability of success if both entities appear in a containment context
                return 0.9
    
    # D. Cleaning an object
    # Expected progression: "clean the apple" -> "the apple is clean"
    if "clean" in a:
        obj = a.replace("clean", "").replace("the", "").strip()
        if obj and obj in ns and "is clean" in ns:
            return 0.8

    # 4. General Progress Heuristic
    # If the state text has changed and no failure was detected, 
    # it is likely a minor step toward the goal.
    if ns != s:
        return 0.1
        
    return 0.0