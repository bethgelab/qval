import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given ALFWorld state representation.
    The value is based on proximity to the goal, object manipulation status,
    and current location.
    """
    state_lower = state.lower()
    
    # 1. Immediate success check
    # If the state indicates the task is completed, return 1.0.
    if "success" in state_lower or "task completed" in state_lower:
        return 1.0

    # 2. Extract the goal from the state
    # Typical format: "Goal: Put the apple in the fridge."
    goal_match = re.search(r"goal: (.*?)(?:\.|$)", state_lower)
    if not goal_match:
        return 0.0
    goal_text = goal_match.group(1)

    # 3. Parse goal components: action, object, and target location
    # Common verbs and prepositions used in ALFWorld tasks
    verbs = ['put', 'place', 'move', 'clean', 'find', 'pick', 'get', 'take', 'wash']
    preps = ['in', 'on', 'at', 'to']
    
    action = None
    for v in verbs:
        if v in goal_text:
            action = v
            break
    
    target_loc = None
    obj = None
    
    # Determine if there is a target location (e.g., "in the fridge")
    found_prep = False
    for p in preps:
        # Use spaces to avoid matching parts of words
        pattern = f" {p} "
        if pattern in f" {goal_text} ":
            parts = goal_text.split(pattern)
            if len(parts) >= 2:
                # The target location follows the preposition
                target_loc = parts[1].strip().replace('the ', '').replace('a ', '').replace('an ', '')
                # The object is in the part before the preposition
                obj_part = parts[0]
                if action and action in obj_part:
                    obj_part = obj_part.replace(action, "")
                obj_words = obj_part.split()
                if obj_words:
                    obj = obj_words[-1]
                found_prep = True
                break
    
    # If no preposition is found, attempt to identify the object from the verb phrase
    if not found_prep:
        # Check for patterns like "clean the apple" or "find the apple"
        obj_match = re.search(r"(?:the|a|an)\s+([a-z]+)", goal_text)
        if obj_match:
            obj = obj_match.group(1)
        else:
            # Fallback: find the first non-stopword as the object
            stopwords = {'the', 'a', 'an', 'is', 'to', 'find', 'clean', 'pick', 'up', 'get', 'take', 'your', 'task'}
            goal_words = re.findall(r'\w+', goal_text)
            for w in goal_words:
                if w not in stopwords:
                    obj = w
                    break
    
    if not obj:
        return 0.0

    # 4. Identify environment status
    # Check if the agent is holding the target object
    holding_match = re.search(r"you are holding ([^.]+)", state_lower)
    holding = holding_match.group(1) if holding_match else "nothing"
    is_holding = (obj in holding)

    # 5. Calculate heuristic score based on progress
    score = 0.0
    
    if action in ['clean', 'wash']:
        # Progress for cleaning tasks
        is_dirty = False
        if obj in state_lower:
            obj_idx = state_lower.find(obj)
            # Check a local context window around the object for the word "dirty"
            context = state_lower[max(0, obj_idx - 20):obj_idx + 20]
            if "dirty" in context:
                is_dirty = True
        
        if is_holding:
            score = 0.8 if not is_dirty else 0.5
        elif not is_dirty and obj in state_lower:
            score = 0.9 # Object is already clean
        elif is_dirty and obj in state_lower:
            score = 0.4 # Object is in room but still dirty
        else:
            score = 0.1 # Object not found
            
    elif target_loc:
        # Progress for movement/placement tasks (e.g., "put apple in fridge")
        # In ALFWorld, items in the state description are typically in the current room.
        is_at_target_loc = (target_loc in state_lower)
        
        # Heuristic to see if the object is already at the target location
        obj_at_target = False
        if obj in state_lower and target_loc in state_lower:
            obj_idx = state_lower.find(obj)
            context = state_lower[max(0, obj_idx - 30):obj_idx + 30]
            if target_loc in context:
                obj_at_target = True

        if obj_at_target:
            score = 0.9
        elif is_holding and is_at_target_loc:
            score = 0.9
        elif is_holding:
            score = 0.6
        elif is_at_target_loc:
            score = 0.3
        elif obj in state_lower:
            score = 0.2
        else:
            score = 0.1
    else:
        # Progress for simple retrieval/search tasks (e.g., "find apple")
        if is_holding:
            score = 0.9
        elif obj in state_lower:
            score = 0.5
        else:
            score = 0.2

    return min(float(score), 1.0)