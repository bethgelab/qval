import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is higher when the agent is closer to completing its task.
    """
    # 1. Parse the Task from the state string
    # ALFWorld typically includes the instruction: "Your task is to [action] [object] [preposition] [location]"
    task_match = re.search(r"task is to\s+(.*?)(?:\.|$)", state, re.IGNORECASE)
    if not task_match:
        return 0.0
    
    task = task_match.group(1).lower()
    
    # 2. Identify Action, Target Object, and Target Location
    # Common ALFWorld actions: put, move, place, clean, find, grab, pick
    action = ""
    for a in ["put", "move", "place", "clean", "find", "grab", "pick"]:
        if a in task:
            action = a
            break
            
    target_obj = None
    target_loc = None
    
    # Attempt to extract components using regex patterns
    if action in ["put", "move", "place"]:
        # Looking for pattern: "action [object] [prep] [location]"
        # e.g., "put apple in fridge" or "move the apple to the table"
        m = re.search(r"(?:put|move|place)\s+(?:the\s+|a\s+|an\s+)?(\w+)\s+(?:in|on|at|to)\s+(?:the\s+)?(\w+)", task)
        if m:
            target_obj = m.group(1)
            target_loc = m.group(2)
    elif action == "clean":
        # e.g., "clean the apple"
        m = re.search(r"clean\s+(?:the\s+|a\s+|an\s+)?(\w+)", task)
        if m:
            target_obj = m.group(1)
    elif action == "find":
        # e.g., "find the apple"
        m = re.search(r"find\s+(?:the\s+|a\s+|an\s+)?(\w+)", task)
        if m:
            target_obj = m.group(1)
            
    # Fallback for task parsing if regex fails
    if not target_obj:
        words = task.split()
        for i, w in enumerate(words):
            if w in ["put", "move", "place", "clean", "find", "grab", "pick"] and i + 1 < len(words):
                target_obj = words[i+1]
                break
                
    if not target_obj:
        return 0.0

    # 3. Analyze Current State
    # Identify Agent Location (the room or container the agent is currently in)
    agent_loc = ""
    loc_match = re.search(r"you are in (?:the\s+)?(\w+)", state, re.IGNORECASE)
    if loc_match:
        agent_loc = loc_match.group(1).lower()
        
    # Check if the target object is in the agent's inventory
    is_in_hand = False
    # Common inventory descriptions: "you are carrying an apple", "you have a knife", "holding an apple"
    if re.search(rf"you (?:are carrying|have|are holding|holding)\s+(?:a\s+|an\s+)?{re.escape(target_obj)}", state, re.IGNORECASE):
        is_in_hand = True
    elif re.search(rf"{re.escape(target_obj)}\s+in your hand", state, re.IGNORECASE):
        is_in_hand = True
        
    # Identify where the object is located
    obj_location = None
    # e.g., "the apple is in the fridge" or "on the table there is an apple"
    obj_loc_match = re.search(rf"{re.escape(target_obj)}\s+(?:is\s+)?(?:in|on|at)\s+(?:the\s+)?(\w+)", state, re.IGNORECASE)
    if obj_loc_match:
        obj_location = obj_loc_match.group(1).lower()
    else:
        obj_loc_match = re.search(rf"(?:in|on|at)\s+(?:the\s+)?(\w+).+?{re.escape(target_obj)}", state, re.IGNORECASE)
        if obj_loc_match:
            obj_location = obj_loc_match.group(1).lower()

    # 4. Scoring Logic based on task progress
    if action in ["put", "move", "place"]:
        if target_loc and target_obj:
            # Goal achieved: Object is at destination
            if obj_location == target_loc.lower():
                return 1.0
            # Near completion: Agent is at destination and holds the object
            if agent_loc == target_loc.lower() and is_in_hand:
                return 0.9
            # Progress: Agent is holding the object
            if is_in_hand:
                return 0.6
            # Progress: Object is found in the environment
            if target_obj in state.lower():
                return 0.3
        return 0.0
        
    elif action == "clean":
        # Goal achieved: Object is clean
        if re.search(rf"{re.escape(target_obj)}\s+is\s+clean", state, re.IGNORECASE):
            return 1.0
        # Progress: Agent is holding the object
        if is_in_hand:
            return 0.6
        # Progress: Object is in the current room/visible
        if target_obj in state.lower():
            return 0.3
        return 0.0
        
    elif action == "find":
        # Goal achieved: Object is in hand
        if is_in_hand:
            return 1.0
        # Progress: Object is visible in the room
        if target_obj in state.lower():
            return 0.5
        return 0.1

    # Final fallback for unknown actions or missing data
    if target_obj in state.lower():
        return 0.2
    return 0.0