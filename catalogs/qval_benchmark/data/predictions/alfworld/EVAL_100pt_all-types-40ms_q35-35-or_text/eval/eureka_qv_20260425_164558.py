def signal_function(state: str, action: str, next_state: str):
    success_bonus = 0.0
    progress_bonus = 0.0
    stagnation_penalty = 0.0
    error_penalty = 0.0
    goal_progress_bonus = 0.0
    efficiency_bonus = 0.0
    manipulation_bonus = 0.0
    navigation_bonus = 0.0
    
    next_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # 1. Check for terminal success - expanded keywords and patterns
    success_patterns = [
        "success!", "success", "task complete", "task completed", 
        "goal achieved", "mission accomplished", "well done",
        "congratulations", "correct", "you did it", "task finished",
        "you have completed", "all done", "complete", "done",
        "you succeeded", "you found", "you placed", "you moved",
        "the object is", "the goal is", "task is", "done successfully",
        "mission complete", "task success", "completed successfully",
        "the task is complete", "you have successfully", "task is done",
        "success!", "episode complete", "task succeeded"
    ]
    for kw in success_patterns:
        if kw in next_lower:
            success_bonus = 1.0
            return 1.0, {
                "success_bonus": 1.0,
                "progress_bonus": 0.0,
                "stagnation_penalty": 0.0,
                "error_penalty": 0.0,
                "goal_progress_bonus": 0.0,
                "efficiency_bonus": 0.0,
                "manipulation_bonus": 0.0,
                "navigation_bonus": 0.0,
            }
    
    # 2. Check for action errors/failures - granular by error type
    error_keywords = {
        "navigation": ["cannot", "can't go", "can't move", "blocked", "obstructed", "no path", "unreachable", "not possible"],
        "manipulation": ["nothing", "not found", "doesn't work", "broken", "can't pick", "can't take", "can't grab", "no such thing", "nothing there", "no item"],
        "state_change": ["isn't", "is not", "can't open", "can't close", "can't turn", "can't wash", "can't fill", "already", "is already", "isn't"],
        "general": ["invalid", "error", "failed", "impossible", "try again", "try to", "doesn't seem", "wrong", "incorrect"]
    }
    
    error_type = "general"
    error_count = 0
    for etype, keywords in error_keywords.items():
        count = sum(1 for kw in keywords if kw in next_lower)
        if count > 0:
            error_count += count
            if etype == "navigation":
                error_penalty = -0.15 * (1 + 0.15 * count)
            elif etype == "manipulation":
                error_penalty = -0.20 * (1 + 0.20 * count)
            elif etype == "state_change":
                error_penalty = -0.18 * (1 + 0.18 * count)
            else:
                error_penalty = -0.12 * (1 + 0.12 * count)
    
    error_penalty = max(-0.60, error_penalty)
    
    # 3. Check for stagnation - state didn't change
    if state == next_state:
        stagnation_penalty = -0.40
    elif state_lower == next_lower:
        stagnation_penalty = -0.22
    
    # 4. Stage-based progress analysis
    search_indicators = ["search", "look", "find", "located", "near", "there is", "is there", "you see", "looked", "searched"]
    search_matches = sum(1 for kw in search_indicators if kw in next_lower)
    
    manipulation_indicators = [
        "picked", "taken", "holding", "carried", "grabbed", 
        "in my hand", "picked up", "in hand", "has",
        "opened", "closed", "turned on", "turned off", "cleaned",
        "washed", "dried", "filled", "emptied", "placed", "put",
        "moved", "carried", "grabbed", "taken from", "picked up from",
        "holding", "on my", "in my"
    ]
    manipulation_matches = sum(1 for kw in manipulation_indicators if kw in next_lower)
    
    placement_indicators = [
        "placed", "put", "on ", "in ", "at ", "onto", "into", 
        "to the", "on the", "in the", "at the", "on the ", "in the ",
        "placed on", "placed in", "put on", "put in", "put to",
        "goes", "goes to", "on the ", "in the "
    ]
    placement_matches = sum(1 for kw in placement_indicators if kw in next_lower)
    
    # 5. Calculate progress_bonus with better scaling
    if manipulation_matches > 0 and placement_matches > 0:
        progress_bonus = 0.25 + 0.08 * manipulation_matches + 0.12 * placement_matches
    elif manipulation_matches > 0 and placement_matches == 0:
        progress_bonus = 0.18 + 0.10 * manipulation_matches
    elif search_matches > 0 and manipulation_matches == 0 and placement_matches == 0:
        progress_bonus = 0.06 + 0.03 * search_matches
    else:
        progress_bonus = 0.02
    
    progress_bonus = min(0.60, progress_bonus)
    
    # 6. Goal_progress_bonus with enhanced location and object tracking
    goal_location_indicators = {
        "primary": ["bedroom", "kitchen", "living room", "dining room", "bathroom"],
        "secondary": ["countertop", "table", "desk", "sink", "toilet", "bathtub", "floor"],
        "tertiary": ["trashcan", "garbage", "cupboard", "cabinet", "drawer", "shelf", "nightstand"]
    }
    
    location_matches = 0
    for category, keywords in goal_location_indicators.items():
        matches = sum(1 for kw in keywords if kw in next_lower)
        if category == "primary":
            location_matches += matches * 1.2
        elif category == "secondary":
            location_matches += matches * 0.8
        else:
            location_matches += matches * 0.4
    
    object_indicators = ["object", "item", "thing", "item on", "on ", "in "]
    object_matches = sum(1 for kw in object_indicators if kw in next_lower)
    
    # Scale goal progress based on action alignment and location relevance
    if manipulation_matches > 0 or placement_matches > 0:
        if placement_matches > 0:
            goal_progress_bonus = 0.12 + 0.06 * location_matches + 0.04 * object_matches
        elif manipulation_matches > 0:
            goal_progress_bonus = 0.08 + 0.05 * location_matches + 0.03 * object_matches
    elif search_matches > 0:
        goal_progress_bonus = 0.04 + 0.02 * location_matches + 0.01 * object_matches
    else:
        goal_progress_bonus = 0.015
    
    goal_progress_bonus = min(0.45, goal_progress_bonus)
    
    # 7. Navigation bonus - reward purposeful movement
    navigation_indicators = ["go to", "walk to", "move to", "head to", "navigate", "go ", "walk "]
    has_navigation = any(kw in action_lower for kw in navigation_indicators)
    if has_navigation:
        if progress_bonus >= 0.12:
            navigation_bonus = 0.05 + 0.015 * min(progress_bonus, 0.5)
        else:
            navigation_bonus = 0.025
    
    # 8. Manipulation bonus - reward when manipulation leads to placement
    valid_manipulation_actions = [
        "go to", "walk to", "pick", "take", "put", "place", 
        "move", "navigate", "find", "open", "close", "clean",
        "turn on", "turn off", "wash", "dry", "fill", "empty", "grab", "grabbed"
    ]
    has_valid_action = any(kw in action_lower for kw in valid_manipulation_actions)
    
    if has_valid_action and manipulation_matches > 0 and placement_matches > 0:
        manipulation_bonus = 0.07 + 0.03 * min(manipulation_matches, 2)
    elif has_valid_action and manipulation_matches > 0:
        manipulation_bonus = 0.035
    
    # 9. Efficiency bonus - reward fewer steps to goal (based on progress rate)
    efficiency_bonus = 0.015 * min(progress_bonus, 0.5)
    
    # 10. Calculate total
    total = success_bonus + progress_bonus + stagnation_penalty + error_penalty + goal_progress_bonus + efficiency_bonus + manipulation_bonus + navigation_bonus
    
    # Clamp to reasonable range
    total = max(-0.70, min(0.90, total))
    
    return total, {
        "success_bonus": success_bonus,
        "progress_bonus": progress_bonus,
        "stagnation_penalty": stagnation_penalty,
        "error_penalty": error_penalty,
        "goal_progress_bonus": goal_progress_bonus,
        "efficiency_bonus": efficiency_bonus,
        "manipulation_bonus": manipulation_bonus,
        "navigation_bonus": navigation_bonus,
    }