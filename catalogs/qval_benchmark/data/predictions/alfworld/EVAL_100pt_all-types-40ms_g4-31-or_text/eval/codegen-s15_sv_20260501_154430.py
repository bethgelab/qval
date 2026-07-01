import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given state in an ALFWorld environment.
    The value represents the expected discounted cumulative reward, effectively 
    acting as the probability of successfully completing the task.
    """
    s = state.lower()
    
    # 1. Immediate Success: Check for terminal success indicators
    if any(x in s for x in ["success", "task completed", "goal achieved", "successfully put"]):
        return 1.0
    
    # 2. Target Extraction: Identify the target object and destination from the goal
    target_obj = None
    destination = None
    is_cleaning_task = False
    
    # Common ALFWorld goal patterns: 
    # "put the [object] in the [destination] [in the room]"
    put_match = re.search(r"put the (.*?) in the (.*?)(?: in the|$|\.)", s)
    if put_match:
        target_obj = put_match.group(1).strip()
        destination = put_match.group(2).strip()
    
    # "clean the [object]"
    clean_match = re.search(r"clean the (.*?)(?:\.|$)", s)
    if clean_match:
        # Only set target_obj if not already found or if this is a primary objective
        if not target_obj:
            target_obj = clean_match.group(1).strip()
        is_cleaning_task = True

    # If we cannot identify the target object from the text, we return a baseline exploration value
    if not target_obj:
        return 0.1
        
    # Check if target object is already at the destination
    if destination and f"{target_obj} is in the {destination}" in s:
        return 1.0

    # 3. Evaluation based on Progress
    # Possession: The agent has picked up the target object
    holding = False
    if f"holding the {target_obj}" in s or f"holding {target_obj}" in s:
        holding = True
    elif "holding" in s:
        # Flexibly check if target_obj is part of the holding phrase
        holding_match = re.search(r"holding (?:the )?(.*)", s)
        if holding_match and target_obj in holding_match.group(1):
            holding = True

    if holding:
        # Possession is a strong signal of progress
        val = 0.6
        
        # For cleaning tasks, check if the item has been cleaned while held
        if is_cleaning_task:
            if f"{target_obj} is clean" in s or "is now clean" in s or "cleaned the" in s:
                val = 0.9
            else:
                val = 0.4 # Held but not yet cleaned
        
        # For putting tasks, check if agent is at the destination
        if destination:
            # If the destination is visible or agent is 'at' it, value increases significantly
            if f"see {destination}" in s or f"see the {destination}" in s or f"at the {destination}" in s:
                val = max(val, 0.9)
        return val
        
    # 4. Proximity: Target object or destination is in the current environment
    # The agent sees the target object (prepared to pick it up)
    if f"see {target_obj}" in s or f"see the {target_obj}" in s or f"{target_obj} is" in s:
        return 0.4
        
    # The agent sees the destination (prepared to place the object)
    if destination and (f"see {destination}" in s or f"see the {destination}" in s or f"{destination} is" in s):
        return 0.3
        
    # Baseline value for initial states or exploration phases
    return 0.1