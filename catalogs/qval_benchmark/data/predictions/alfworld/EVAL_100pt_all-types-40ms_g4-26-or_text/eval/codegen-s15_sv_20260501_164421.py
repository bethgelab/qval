import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given ALFWorld state description.
    The value is based on identifying the goal from the instruction
    and checking the progress of that goal in the observation.
    """
    s = state.lower()
    
    # 1. Check for immediate success or terminal states
    # If the text indicates the task is completed, the value is 1.0.
    success_indicators = ["success", "completed", "done", "achieved"]
    if any(indicator in s for indicator in success_indicators):
        # Verification that this is a terminal state rather than just an instruction
        if any(term in s for term in ["task", "successfully", "completed", "achieved"]):
            return 1.0

    # 2. Extraction of Instruction and Observation
    # ALFWorld states often follow the format: "Instruction: [task]. Observation: [description]."
    instr = ""
    obs = s
    
    # Try to isolate the instruction part using regex
    instr_match = re.search(r"(?:instruction|goal):\s*(.*?)(?:\.|\n|observation:|$)", s)
    if instr_match:
        instr = instr_match.group(1).strip()
        # Try to isolate the observation part
        obs_match = re.search(r"observation:\s*(.*)", s, re.DOTALL)
        if obs_match:
            obs = obs_match.group(1).strip()
    else:
        # If no explicit instruction/goal prefix is found, we assume the context
        # is part of the state, but parsing might be less accurate.
        instr = s
        obs = s

    # 3. Parsing the Instruction to identify the goal
    # We look for common action patterns: put, pick up, clean, open.
    obj, loc, action = None, None, None
    
    # Regex patterns for task components
    # pattern: "put the apple in the fridge" -> obj="apple", loc="fridge"
    put_m = re.search(r"put\s+(?:the\s+)?(\w+)\s+(?:in|on|at)\s+(?:the\s+)?(\w+)", instr)
    # pattern: "pick up the apple" -> obj="apple"
    pick_m = re.search(r"pick\s+up\s+(?:the\s+)?(\w+)", instr)
    # pattern: "clean the apple" -> obj="apple"
    clean_m = re.search(r"clean\s+(?:the\s+)?(\w+)", instr)
    # pattern: "open the fridge" -> obj="fridge"
    open_m = re.search(r"open\s+(?:the\s+)?(\w+)", instr)

    if put_m:
        action, obj, loc = "put", put_m.group(1), put_m.group(2)
    elif pick_m:
        action, obj = "pick", pick_m.group(1)
    elif clean_m:
        action, obj = "clean", clean_m.group(1)
    elif open_m:
        action, obj = "open", open_m.group(1)
    
    # If no specific goal can be parsed, return a low baseline value.
    if not obj:
        return 0.0

    # 4. Progress Estimation
    # We check the observation for predicates that indicate task progression.
    # Example: Is the object being held? Is the object in the target location?
    
    # Check if the object is currently held by the agent
    is_held = re.search(rf"holding\s+(?:the\s+)?{obj}|{obj}\s+is\s+in\s+your\s+hand", obs)
    
    if action == "put":
        # Goal: object in/on/at location
        if re.search(rf"{obj}.*?(?:is|are|placed|in|on|at).*?{loc}", obs):
            return 1.0
        if is_held:
            return 0.7
        if re.search(rf"{obj}.*?(?:is|are|on|in|at)", obs):
            return 0.3
            
    elif action == "pick":
        # Goal: object in hand
        if is_held:
            return 1.0
        if re.search(rf"{obj}.*?(?:is|are|on|in|at)", obs):
            return 0.4
            
    elif action == "clean":
        # Goal: object is clean
        if re.search(rf"{obj}.*?is\s+clean", obs):
            return 1.0
        if re.search(rf"{obj}.*?(?:is|are|on|in|at)", obs):
            return 0.4
            
    elif action == "open":
        # Goal: object is open
        if re.search(rf"{obj}.*?is\s+open", obs):
            return 1.0
        if re.search(rf"{obj}.*?(?:is|are|on|in|at)", obs):
            return 0.4

    # Default return value if the task is identified but no progress is seen.
    return 0.1