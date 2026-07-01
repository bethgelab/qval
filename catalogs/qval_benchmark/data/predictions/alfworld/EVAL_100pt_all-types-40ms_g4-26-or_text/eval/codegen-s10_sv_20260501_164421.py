import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given ALFWorld state.
    The value is based on progress towards the goal extracted from the state text.
    """
    s = state.lower()
    
    # 1. Check for terminal success
    # If the environment indicates the task is complete, the value is 1.0.
    if any(kw in s for kw in ["task is completed", "successfully", "finished", "completed"]):
        return 1.0
    
    # 2. Identify the goal string
    # ALFWorld states usually contain an instruction like "The goal is to..."
    goal_str = ""
    goal_indicators = ["goal is to", "task is to", "the goal is to", "you must", "your task is to"]
    for ind in goal_indicators:
        if ind in s:
            idx = s.find(ind)
            goal_str = s[idx:]
            break
    
    if not goal_str:
        # If no explicit goal instruction is found, we cannot accurately estimate value.
        return 0.0
        
    # 3. Parse the goal and estimate progress
    # Pattern 1: Complex goal (action + object + preposition + location)
    # e.g., "put the apple in the microwave"
    complex_pattern = r"(?:put|place|move|clean|find|drop|get|pick up) (?:the )?(.*?) (in|on|at|inside|to) (?:the )?(.*?)(?:\.|\n|$)"
    match = re.search(complex_pattern, goal_str)
    
    if match:
        obj = match.group(1).strip()
        prep = match.group(2).strip()
        loc = match.group(3).strip().rstrip('.').rstrip('!')
        
        # Check if the target condition is already met (object is in/on/at the location)
        # We look for "apple is in the microwave" or "apple in the microwave"
        if re.search(rf"{re.escape(obj)} (?:is )?(?:in|on|at|inside|to) (?:the )?{re.escape(loc)}", s):
            return 0.8
        
        # Check if the agent is currently holding the target object
        if re.search(rf"holding (?:a|an|the )?{re.escape(obj)}", s):
            return 0.5
        
        # Check if the agent is in the target location (if the location is a room)
        if re.search(rf"you are (?:in|at) (?:the |a )?{re.escape(loc)}", s):
            return 0.3
            
        # Check if both the object and the location are present/visible
        if obj in s and loc in s:
            return 0.2
            
        # Check if the object is at least visible
        if obj in s:
            return 0.1
            
    else:
        # Pattern 2: Simple goal (action + object)
        # e.g., "find the apple" or "clean the apple"
        simple_pattern = r"(?:find|clean|pick up|get|move) (?:the )?(.*?)(?:\.|\n|$)"
        match = re.search(simple_pattern, goal_str)
        if match:
            obj = match.group(1).strip()
            if obj in s:
                # If the agent is holding the object, progress is higher
                if re.search(rf"holding (?:a|an|the )?{re.escape(obj)}", s):
                    return 0.7
                return 0.3
            return 0.0
            
    return 0.0