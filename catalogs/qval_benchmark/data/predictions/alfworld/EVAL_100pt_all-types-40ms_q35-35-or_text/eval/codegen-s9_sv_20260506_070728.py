import re

def signal_function(state: str) -> float:
    s = state.lower()
    
    # Check for terminal success states
    if re.search(r'success|completed|done|task complete|goal achieved', s):
        return 1.0
    
    score = 0.0
    
    # Try to parse the task object from the state description
    # Common ALFWorld task verbs
    task_match = re.search(r'task:\s*(?:put|clean|heat|cool|open|close|wash|fill|empty|unplug|plug)\s+(?:the\s+)?(\w+)', s)
    task_obj = task_match.group(1) if task_match else None
    
    # Try to parse what the agent is holding
    hold_match = re.search(r'holding\s+(?:the\s+)?(\w+)', s)
    held_obj = hold_match.group(1) if hold_match else None
    
    # Evaluate progress based on object manipulation
    if task_obj and held_obj:
        if task_obj == held_obj:
            score = 0.95  # Holding the correct object for the task
        else:
            score = 0.6   # Holding an object, but maybe not the right one
    elif held_obj and held_obj != 'nothing':
        score = 0.6     # Holding something (progress indicator)
    elif task_obj:
        score = 0.4     # Task object is visible in the environment
    else:
        score = 0.2     # Basic state presence
    
    # Bonus for being in a room (proximity to potential locations)
    if re.search(r'you are in the', s):
        score += 0.1
    
    # Bonus for being near common destination surfaces
    if re.search(r'in the microwave|on the countertop|in the sink|in the trashcan', s):
        score += 0.1
        
    return min(1.0, max(0.0, score))