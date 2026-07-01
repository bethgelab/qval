import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld task.
    The value is based on progress toward the goal: finding the object, 
    picking it up, cleaning it (if required), and placing it in the target location.
    """
    if not state:
        return 0.0

    state_lower = state.lower()
    
    # 1. Extract the goal from the state
    # Typical ALFWorld goal format: "Goal: put a clean apple in the fridge"
    goal_match = re.search(r"goal:\s*(.*)", state_lower)
    if not goal_match:
        return 0.0
    
    goal_text = goal_match.group(1).strip()
    
    # 2. Parse target object and target location
    # Pattern: "put [a/the] [object] in [the/a] [location]"
    target_match = re.search(r"put\s+(?:a|the)\s+(.*?)\s+in\s+(?:the|a)\s+(.*)", goal_text)
    if not target_match:
        return 0.0
    
    full_target_obj = target_match.group(1).strip()
    target_loc = target_match.group(2).strip()
    
    # Determine if cleaning is required
    needs_cleaning = "clean" in full_target_obj
    # The actual object name (e.g., "apple" instead of "clean apple")
    target_obj = full_target_obj.replace("clean ", "").strip()
    
    # 3. Analyze current state features
    # Check if agent is holding the object
    is_holding = f"holding {target_obj}" in state_lower
    
    # Check if the object is clean (if cleaning was required)
    # Often represented as "holding clean apple" or "the apple is clean"
    is_clean = False
    if needs_cleaning:
        is_clean = f"clean {target_obj}" in state_lower or f"{target_obj} is clean" in state_lower
    else:
        is_clean = True  # Not required, so treated as satisfied
        
    # Check if the object is visible (but not necessarily held)
    sees_obj = target_obj in state_lower and not is_holding
    
    # Check if the destination location is visible or agent is at the location
    sees_loc = target_loc in state_lower
    
    # Check if the task is likely completed
    # e.g., "the apple is in the fridge"
    completion_pattern = f"{target_obj}.*?in the {target_loc}"
    is_completed = re.search(completion_pattern, state_lower) is not None

    # 4. Assign state-value based on milestones
    if is_completed:
        return 1.0
    
    score = 0.0
    
    if is_holding:
        # Holding the object is a major milestone
        if is_clean:
            score += 0.6
        else:
            score += 0.3  # Still needs cleaning
            
        # Bonus for being at the destination while holding the clean object
        if is_clean and sees_loc:
            score += 0.2
    else:
        # Not holding the object yet
        if sees_obj:
            score += 0.2  # Found the object
        if sees_loc:
            score += 0.1  # In the right area, but object not yet acquired
            
    # Cap the value at 0.9 to ensure terminal success is the maximum
    return min(score, 0.9)