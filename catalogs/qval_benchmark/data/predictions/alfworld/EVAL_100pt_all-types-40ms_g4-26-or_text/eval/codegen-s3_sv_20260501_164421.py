import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given ALFWorld state description.
    The value is based on the task progress towards the goal.
    """
    s = state.lower()
    
    # 1. Extract Goal Information
    # ALFWorld goals are typically formatted as "goal is to [action] the [object] [prep] the [location]"
    goal_match = re.search(r"goal is to (.*?)(?:\.|$)", s)
    if not goal_match:
        return 0.0
    goal = goal_match.group(1)
    
    target_obj = None
    target_loc = None
    task_type = None
    
    # Task type identification and target extraction
    if any(x in goal for x in ["put", "move", "place"]):
        task_type = "move"
        m = re.search(r"(?:put|move|place) the (.*?) (?:in|on|at|to) the (.*)", goal)
        if m:
            target_obj = m.group(1).strip()
            target_loc = m.group(2).strip()
    elif "clean" in goal:
        task_type = "clean"
        m = re.search(r"clean the (.*)", goal)
        if m:
            target_obj = m.group(1).strip()
    elif "find" in goal:
        task_type = "find"
        m = re.search(r"find the (.*)", goal)
        if m:
            target_obj = m.group(1).strip()
            
    if not target_obj:
        return 0.0
        
    # Using the last word of the target object/location to handle descriptions 
    # like "the red apple" vs just "apple"
    obj_key = target_obj.split()[-1]
    
    # 2. Evaluate Success (Reward = 1.0)
    if task_type == "move" and target_loc:
        loc_key = target_loc.split()[-1]
        # Search for patterns indicating the object is at the location (e.g., "the apple is in the fridge")
        if re.search(rf"{re.escape(obj_key)}.*(?:is|located|placed|on|in).*{re.escape(loc_key)}", s):
            return 1.0
    elif task_type == "clean":
        if re.search(rf"{re.escape(obj_key)}.*is clean", s):
            return 1.0
    elif task_type == "find":
        if re.search(rf"found the {re.escape(obj_key)}", s) or re.search(rf"you see the {re.escape(obj_key)}", s):
            return 1.0
    
    # 3. Evaluate Progress (Heuristic state-value)
    # Check if the agent is currently holding the target object
    is_holding = "holding" in s and re.search(rf"{re.escape(obj_key)}", s)
    
    # Identify the current room/location of the agent
    room_match = re.search(r"you are in the ([\w\s]+)", s)
    current_room = room_match.group(1).strip() if room_match else ""
    
    # Check if the agent is in the target location
    in_target_loc = False
    if target_loc and (target_loc in current_room or target_loc.split()[-1] in current_room):
        in_target_loc = True

    # Heuristic scoring logic:
    # - Holding object and in target location (near completion): 0.9
    # - Holding object (midway): 0.7
    # - Object is in the current room (discovered): 0.4
    # - Default/No progress: 0.1
    if is_holding:
        if task_type == "move" and in_target_loc:
            return 0.9
        return 0.7
    
    # If object is in the same room as the agent
    if obj_key in s and current_room and current_room in s:
        return 0.4
    elif obj_key in s and not current_room:
        # If we don't know the room but the object is mentioned
        return 0.2
        
    return 0.1