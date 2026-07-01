import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the ALFWorld environment.
    The Q-value is approximated based on progress towards common task milestones.
    """
    # Normalize inputs for consistent analysis
    s = state.lower()
    a = action.lower()
    ns = next_state.lower()
    
    # Failure indicators: if the action resulted in no state change or a negative response
    failure_keywords = ["cannot", "fail", "invalid", "not found", "not there", "not possible", "not available"]
    is_failure = any(kw in ns for kw in failure_keywords)
    
    # 1. Terminal Success: Highest value
    if "success" in ns or "completed" in ns or "task finished" in ns:
        return 1.0
        
    # 2. Failure or No-Op: Lowest value
    if is_failure or ns == s:
        return 0.0
        
    # 3. Target object extraction
    # ALFWorld tasks often follow patterns like "Put the apple in the fridge" or "Clean the table"
    target_obj = None
    patterns = [
        r"put the (.*?) in the (.*?)\.",
        r"clean the (.*?)\.",
        r"move the (.*?) to the (.*?)\.",
        r"place the (.*?) in the (.*?)\.",
        r"task: (.*?) in the (.*?)\."
    ]
    for p in patterns:
        match = re.search(p, s)
        if match:
            target_obj = match.group(1).strip()
            break
    
    # 4. Progress-based Estimation
    
    # A: Final high-value actions (placing or cleaning the object)
    # These are the most critical steps before completion.
    if ("put" in a or "clean" in a or "place" in a):
        if target_obj and target_obj in a:
            return 0.9
        return 0.8
        
    # B: Possession of the target object
    # Holding the object is a major milestone in almost every ALFWorld task.
    if "holding" in ns:
        return 0.7
        
    # C: Attempting to acquire the object
    # Successful pick-up actions lead directly to the "holding" state.
    if "take" in a or "pick up" in a:
        return 0.6
        
    # D: Interaction with the target object or its location
    # Seeing the object or interacting with the container it is in.
    if target_obj:
        if target_obj in ns:
            return 0.4
        if target_obj in a:
            return 0.3
            
    # E: General exploration and navigation
    # Moving between rooms or opening containers is basic progress.
    if any(cmd in a for cmd in ["go to", "open", "examine", "look"]):
        return 0.2
        
    # Baseline value for any other state-changing action
    return 0.1