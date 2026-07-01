import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value of an ALFWorld state by parsing the goal and 
    comparing it against the current observations.
    """
    s = state.lower()
    
    # 1. Identify the boundary between the observation and the task description
    task_start_idx = -1
    for marker in ["task is to", "your task is to", "goal is to"]:
        idx = s.find(marker)
        if idx != -1:
            task_start_idx = idx
            break
    
    if task_start_idx == -1:
        return 0.0
    
    observation = s[:task_start_idx]
    task_part = s[task_start_idx:]
    
    # 2. Parse the task for the target object and (optionally) the destination
    obj = None
    dest = None
    
    # Pattern for movement tasks: "[action] [object] [preposition] [destination]"
    # e.g., "put the apple in the microwave"
    move_match = re.search(r"(?:put|move|place|go)\s+(?:the\s+)?(\w+)\s+(?:in|on|to|at)\s+(?:the\s+)?(\w+)", task_part)
    if move_match:
        obj, dest = move_match.groups()
    else:
        # Pattern for simple tasks: "[action] [object]"
        # e.g., "clean the apple"
        action_match = re.search(r"(?:clean|find|locate|get|pick)\s+(?:the\s+)?(\w+)", task_part)
        if action_match:
            obj = action_match.group(1)
        else:
            # Fallback: grab the first substantial word as the object
            words = re.findall(r"\b\w{3,}\b", task_part)
            if words:
                obj = words[0]
                
    if not obj:
        return 0.0
        
    # 3. Check if the task is already completed (Value = 1.0)
    if dest:
        # Check for object-destination relationship in the observation
        # Matches "apple is in the microwave", "apple in the microwave", etc.
        if re.search(rf"\b{obj}\b\s+(?:is\s+)?(?:in|on|at)\s+(?:the\s+)?\b{dest}\b", observation):
            return 1.0
    elif "clean" in task_part:
        # Hypothetical check for state-based success
        if f"{obj} is clean" in observation:
            return 1.0
            
    # 4. Heuristic Scoring for non-terminal states
    score = 0.0
    
    # Factor 1: Is the agent holding the required object? (High importance)
    if re.search(rf"holding\s+(?:a|an|the\s+)?\b{obj}\b", observation):
        score += 0.6
    
    # Factor 2: Is the object visible in the current room?
    # If it's in the observation but not being held, the agent is near it.
    if re.search(rf"\b{obj}\b", observation) and not re.search(rf"holding\s+(?:a|an|the\s+)?\b{obj}\b", observation):
        score += 0.2
        
    # Factor 3: Is the destination (object or room) in the current room?
    if dest:
        # If dest is an object (microwave), it will be in the room description.
        # If dest is a room (kitchen), "you are in the kitchen" will be in the description.
        if re.search(rf"\b{dest}\b", observation):
            score += 0.2
            
        # Factor 4: Is the agent already in the target room?
        if re.search(rf"you are in (?:the\s+)?\b{dest}\b", observation):
            score += 0.2
            
    # Return the accumulated score, capped below 1.0 to distinguish from terminal states
    return min(score, 0.95)