import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld state by analyzing
    the textual description for goal completion, object proximity,
    and task-oriented progress.
    """
    s = state.lower()
    
    # 1. Check for explicit terminal success indicators
    success_keywords = ["success", "complete", "finished", "done", "goal reached", "task completed"]
    if any(k in s for k in success_keywords):
        return 1.0
        
    # 2. Attempt to extract the goal from the state description
    # ALFWorld states often include the goal in the format: "The goal is to [task]."
    goal_match = re.search(r"goal is to (.*?)(?:\.|$)", s)
    if not goal_match:
        # If no explicit goal is found in the current observation, return a baseline value
        return 0.05
        
    goal = goal_match.group(1)
    
    # 3. Extract target components from the goal
    # We look for nouns following articles (the/a/an) to identify the object and location
    # Example: "put the apple in the fridge" -> ['apple', 'fridge']
    # Example: "clean the apple" -> ['apple']
    targets = re.findall(r"(?:the|a|an)\s+(\w+)", goal)
    
    if not targets:
        return 0.05
        
    target_obj = targets[0]
    target_loc = targets[1] if len(targets) > 1 else None
    
    # Base value for having found a goal context
    score = 0.1
    
    # 4. Evaluate progress toward the identified goal
    # Check if the target object is mentioned in the current state
    obj_present = bool(re.search(rf"\b{target_obj}\b", s))
    
    if obj_present:
        score += 0.2
        
        # Check if the agent is currently holding the target object
        if re.search(rf"holding\s+(?:the|a|an)\s+{target_obj}", s):
            score += 0.3
            
        # Handle attribute-based goals (e.g., "clean the apple")
        if "clean" in goal:
            # Check if the object is described as clean
            if re.search(rf"\b{target_obj}\b.*?\bclean\b", s) or \
               re.search(rf"\bclean\b.*?\b{target_obj}\b", s):
                score += 0.5
                
        # Handle location-based goals (e.g., "put the apple in the fridge")
        if target_loc:
            loc_present = bool(re.search(rf"\b{target_loc}\b", s))
            if loc_present:
                score += 0.2
                
                # Check if the object and location are semantically linked in the state
                # e.g., "the apple is in the fridge" or "the fridge contains an apple"
                spatial_pattern = rf"\b{target_obj}\b.*?\b(?:in|on|at|inside|to)\b\s+(?:the|a|an)\s+\b{target_loc}\b"
                reverse_spatial_pattern = rf"\b(?:in|on|at|inside|to)\b\s+(?:the|a|an)\s+\b{target_loc}\b.*?\b{target_obj}\b"
                
                if re.search(spatial_pattern, s) or re.search(reverse_spatial_pattern, s):
                    score += 0.5
                
                # Check if the agent is in the room where the target location is
                if re.search(rf"you are in the\s+{target_loc}", s):
                    score += 0.1
                    
    # 5. Ensure the returned value is within the valid [0, 1] range
    return max(0.0, min(score, 1.0))