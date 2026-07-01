import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment.
    The value is an approximation of the probability of success, grounded in 
    the progress made toward the goal (locating object -> holding object -> placing object).
    """
    s = state.lower()
    
    # 1. Attempt to extract the goal from the state string.
    # Typical goals: "put a clean apple in the fridge", "put the apple in the microwave"
    # Pattern captures (cleaning requirement, object name, target location)
    goal_pattern = r"put\s+(?:a\s+|the\s+)(?:(clean)\s+)?([\w\s]+?)\s+in\s+(?:the\s+)?([\w\s]+)"
    goal_match = re.search(goal_pattern, s)
    
    if not goal_match:
        # If no clear goal is found in the text, we check for success or return base value.
        if "success" in s or "task completed" in s:
            return 1.0
        return 0.1

    needs_cleaning = goal_match.group(1) == "clean"
    obj_name = goal_match.group(2).strip()
    target_loc = goal_match.group(3).strip()
    
    # 2. Terminal Success check
    # In many ALFWorld states, success is indicated by specific keywords.
    if "success" in s or "task completed" in s:
        return 1.0
    
    # 3. Holding status
    # Check if the agent is currently holding the target object.
    is_holding = False
    if f"holding {obj_name}" in s or f"holding the {obj_name}" in s:
        is_holding = True
    
    # Determine if the object held is 'dirty' when it needs to be 'clean'.
    is_dirty = "dirty" in s and obj_name in s
    
    # 4. Location status
    # Check if the agent is currently at or near the target location.
    is_at_loc = False
    if target_loc in s:
        # Look for proximity indicators like "at the fridge", "in the kitchen", "facing the fridge".
        if re.search(rf"(at|in|near|facing)\s+(the\s+)?{re.escape(target_loc)}", s):
            is_at_loc = True

    # Logic for value assignment:
    
    if is_holding:
        # If the object is dirty and must be clean, the agent still needs to find a sink.
        if needs_cleaning and is_dirty:
            return 0.5 
        # If the agent is holding the clean object and is at the target location.
        if is_at_loc:
            return 0.95
        # Agent has the object but is not yet at the destination.
        return 0.8
        
    if obj_name in s:
        # The object has been spotted or its location is known.
        if "you see" in s or "is in" in s or "on the" in s:
            return 0.6
        return 0.4
        
    if is_at_loc:
        # Agent is at the destination but hasn't found the object yet.
        return 0.2
        
    # Base value for early exploration/search states.
    return 0.1