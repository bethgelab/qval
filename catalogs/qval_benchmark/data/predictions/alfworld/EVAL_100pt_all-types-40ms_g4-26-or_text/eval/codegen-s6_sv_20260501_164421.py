import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is an approximation of the probability of reaching the goal.
    """
    # 1. Identify the goal from the state string
    goal_match = re.search(r"Goal:\s*(.*)", state, re.IGNORECASE)
    if not goal_match:
        return 0.0
    
    goal_text = goal_match.group(1).lower()
    
    # 2. Extract core entities (object and location)
    # We filter out common stopwords and task-related verbs to find the target nouns.
    stop_words = {
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'put', 'place', 'move', 
        'clean', 'find', 'take', 'pick', 'up', 'drop', 'get', 'wash', 
        'wipe', 'with', 'is', 'into', 'and', 'of', 'for', 'by', 'it'
    }
    
    goal_words = re.findall(r'\w+', goal_text)
    core_words = [w for w in goal_words if w not in stop_words]
    
    if not core_words:
        return 0.0
        
    # Typically, the last word is the location and the word before it is the object.
    if len(core_words) >= 2:
        obj_pattern = re.escape(core_words[-2])
        loc_pattern = re.escape(core_words[-1])
        has_loc = True
    else:
        obj_pattern = re.escape(core_words[0])
        loc_pattern = None
        has_loc = False

    # 3. Check for terminal success (Goal reached)
    # Check if the object is in the correct location or has achieved the required state (e.g., cleaned).
    if has_loc:
        # Does the state say the object is in/on/at the location?
        if re.search(rf"{obj_pattern}.*?\b(in|on|at)\b.*?\b{loc_pattern}\b", state, re.IGNORECASE) or \
           re.search(rf"{loc_pattern}.*?\b(in|on|at)\b.*?\b{obj_pattern}\b", state, re.IGNORECASE):
            return 1.0
        # If the task is to clean the object, check if it's clean
        if 'clean' in goal_words and re.search(rf"{obj_pattern}.*?\b(is|was|looks)\b.*?clean", state, re.IGNORECASE):
            return 1.0
    else:
        # If no location is specified, check for state-based success (e.g., "clean the apple")
        if 'clean' in goal_words and re.search(rf"{obj_pattern}.*?\b(is|was|looks)\b.*?clean", state, re.IGNORECASE):
            return 1.0
        # For simple pick/find tasks, being in the possession of the object counts as success
        if ('find' in goal_words or 'take' in goal_words or 'get' in goal_words) and \
           re.search(rf"holding.*?{obj_pattern}", state, re.IGNORECASE):
            return 1.0

    # 4. Estimate progress (Non-terminal state value)
    # We assign scores based on how many goal conditions are partially met.
    score = 0.0
    
    # Condition: Agent is holding the target object
    if re.search(rf"holding.*?{obj_pattern}", state, re.IGNORECASE):
        score += 0.5
        
    # Condition: Target object is visible in the current room/observation
    if re.search(rf"\b{obj_pattern}\b", state, re.IGNORECASE):
        score += 0.2
        
    # Condition: Agent is in the target room/location
    if has_loc and re.search(rf"\b{loc_pattern}\b", state, re.IGNORECASE):
        score += 0.2
        
    # Condition: Agent is holding the object and is in the target room (highly favorable)
    if has_loc and re.search(rf"holding.*?{obj_pattern}", state, re.IGNORECASE) and \
       re.search(rf"\b{loc_pattern}\b", state, re.IGNORECASE):
        score += 0.2
        
    # Condition: For 'find' tasks, seeing the object is significant progress
    if 'find' in goal_words and re.search(rf"\b{obj_pattern}\b", state, re.IGNORECASE):
        score += 0.1

    # Cap the score to ensure it does not exceed 1.0
    return min(score, 0.95)