import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given ALFWorld state.
    The value is based on progress toward the goal extracted from the state text.
    """
    s = state.lower()
    
    # 1. Terminal success check
    # If the environment explicitly states the task is complete.
    success_indicators = ["success", "completed", "finished", "goal reached", "task completed", "you have achieved"]
    if any(ind in s for ind in success_indicators):
        return 1.0
    
    # 2. Goal extraction
    # ALFWorld states often contain the goal in the format: "Your goal is to [task]"
    goal_match = re.search(r"(?:goal is to|task is to|objective is to|your goal is to) (.*?)(?:\.|$)", s)
    if not goal_match:
        # If the goal is not explicitly in the current state observation, 
        # we can't reliably estimate progress.
        return 0.0
    
    goal_text = goal_match.group(1)
    # Extract meaningful keywords by removing common stop words and verbs
    goal_words = re.findall(r'\b\w+\b', goal_text)
    ignore = {
        'to', 'the', 'a', 'an', 'in', 'on', 'at', 'with', 'is', 'of', 'put', 
        'pick', 'up', 'find', 'clean', 'move', 'your', 'it', 'them', 'an', 'be'
    }
    keywords = [w for w in goal_words if w not in ignore]
    
    if not keywords:
        return 0.0
    
    # 3. Progress Heuristics
    # Check which goal components (objects/locations) are present in the current state
    present_keywords = [w for w in keywords if w in s]
    if not present_keywords:
        return 0.0
        
    # Check if the agent is currently 'holding' a key object
    holding = False
    for kw in keywords:
        if f"holding {kw}" in s or f"you are holding {kw}" in s or f"you have {kw}" in s:
            holding = True
            break
    
    # Check if a key relationship is satisfied (e.g., "apple is in fridge" or "apple is clean")
    rel_found = False
    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i == j:
                continue
            # Pattern: Object and Target/State relationship (e.g., "apple is in fridge")
            if re.search(rf"{kw1}.*(?:is|placed|on|in|at|inside|located).*{kw2}", s):
                rel_found = True
                break
            # Pattern: Simple state adjective (e.g., "apple is clean")
            if re.search(rf"{kw1}\s+is\s+{kw2}", s):
                rel_found = True
                break
        if rel_found:
            break
            
    # Check if the agent is in the required room (if the room is mentioned in the goal)
    in_room = False
    for kw in keywords:
        if f"you are in {kw}" in s or f"you are in the {kw}" in s:
            in_room = True
            break
            
    # 4. Value Mapping
    # Mapping qualitative progress to quantitative value
    if rel_found:
        # If the goal condition is met, value is high.
        # If we are also holding the object, we are likely at the exact terminal transition.
        return 0.98 if holding else 0.9
    
    if holding:
        # Being in possession of the target object is a strong indicator of progress.
        return 0.6
    
    if in_room:
        # Being in the correct location is a baseline for starting the task.
        return 0.3
    
    # If we haven't met the high-level progress markers, provide a small value
    # based on how many goal-related words we have encountered in the environment.
    base_progress = 0.1 + (0.2 * (len(present_keywords) / len(keywords)))
    return min(base_progress, 0.5)