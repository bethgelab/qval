import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given ALFWorld state.
    The value reflects the progress toward completing the goal, 
    where 1.0 is success and lower values indicate early stages of the task.
    """
    if not state:
        return 0.0

    state_lower = state.lower()
    
    # 1. Direct Success Detection
    # ALFWorld typically signals success with specific phrases.
    success_keywords = ["task completed", "successfully placed", "successfully cleaned", "mission accomplished"]
    if any(kw in state_lower for kw in success_keywords):
        return 1.0

    # 2. Goal Extraction
    # We look for common ALFWorld goal patterns to identify the target object and location.
    # Patterns: "put a apple in a fridge", "clean the apple", "pick up the apple"
    obj = None
    loc = None
    
    put_match = re.search(r'put (?:a|the) (.+?) in (?:a|the) (.+?)(?:\.|\n|$)', state_lower)
    clean_match = re.search(r'clean (?:a|the) (.+?)(?:\.|\n|$)', state_lower)
    pick_match = re.search(r'pick up (?:a|the) (.+?)(?:\.|\n|$)', state_lower)

    if put_match:
        obj = put_match.group(1).strip()
        loc = put_match.group(2).strip()
    elif clean_match:
        obj = clean_match.group(1).strip()
        loc = "sink"  # Cleaning tasks usually involve a sink or similar utility
    elif pick_match:
        obj = pick_match.group(1).strip()
        loc = None

    # 3. State-Value Estimation based on Progress
    if obj:
        # Indicator: Agent is holding the target object
        # Phrases like "you are holding the apple" or "apple is in your hand"
        holding_obj = (
            f"holding {obj}" in state_lower or 
            f"have {obj}" in state_lower or 
            f"{obj} is in your hand" in state_lower or
            f"holding a {obj}" in state_lower or
            f"holding the {obj}" in state_lower
        )
        
        # Indicator: Target object is visible in the current observation
        see_obj = (
            f"see {obj}" in state_lower or 
            f"{obj} is" in state_lower or 
            f"there is a {obj}" in state_lower
        )
        
        # Indicator: Target location is visible or agent is at the location
        see_loc = False
        if loc:
            see_loc = (
                f"see {loc}" in state_lower or 
                f"{loc} is" in state_lower or 
                f"there is a {loc}" in state_lower or
                f"at the {loc}" in state_lower
            )

        # Progression Scoring Logic
        # Most favorable: Holding object and at target location
        if holding_obj and see_loc:
            return 0.85
        
        # High value: Agent has the object, now needs to find/reach the location
        if holding_obj:
            # If it was a "pick up" task, holding the object is effectively success
            if loc is None:
                return 1.0
            return 0.6
        
        # Medium value: Object and location are both in sight, or just the object is found
        if see_obj and see_loc:
            return 0.4
        if see_obj:
            return 0.3
        
        # Low value: Only the destination is found, or we are just starting
        if see_loc:
            return 0.2

    # Baseline value for states where goal isn't explicitly identified or progress is minimal
    return 0.1