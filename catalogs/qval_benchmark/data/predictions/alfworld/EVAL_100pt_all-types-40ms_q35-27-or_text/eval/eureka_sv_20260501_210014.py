def signal_function(state: str):
    import re
    
    if not state:
        return 0.0, {"success_prob": 0.0, "progress": 0.0, "penalty": 0.0}
    
    s = state.lower()
    
    # Check for task completion - highest priority
    if "task completed" in s or "success" in s or "goal achieved" in s:
        return 1.0, {"success_prob": 1.0, "progress": 0.0, "penalty": 0.0}
    
    # Error states - minimal penalty for early exploration
    error_indicators = ["you cannot", "nothing happens", "there is no", "cannot", "is not", "you are not holding"]
    penalty = -0.04 * sum(1 for ind in error_indicators if ind in s)
    
    # Goal context detection
    goal_keywords = ["put", "move", "clean", "take", "get", "bring"]
    has_goal = any(kw in s for kw in goal_keywords)
    
    # Extract target object from goal with improved regex
    target_object = None
    if has_goal:
        # More flexible pattern for target object
        match = re.search(r'(?:put|move|take|get|bring|clean)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', s)
        if match:
            target_object = match.group(1)
        else:
            # Try alternative pattern
            match = re.search(r'(?:put|move|take|get|bring|clean)\s+(?:a\s+)?(\w+(?:\s+\w+)?)', s)
            if match:
                target_object = match.group(1)
    
    # Extract target location with improved regex
    target_location = None
    if has_goal:
        loc_patterns = [
            r'(?:in|on|into|onto)\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
            r'(?:in|on|into|onto)\s+(?:a\s+)?(\w+(?:\s+\w+)?)',
            r'(?:to\s+)?(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:counter|table|shelf|drawer|cabinet|microwave|fridge|toaster|sink|stove|oven)',
        ]
        for pattern in loc_patterns:
            loc_match = re.search(pattern, s)
            if loc_match:
                target_location = loc_match.group(1)
                break
    
    # What is being held
    is_holding = "you are holding" in s
    holding_object = None
    if is_holding:
        hold_patterns = [
            r'you are holding\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
            r'you are holding\s+(?:a\s+)?(\w+(?:\s+\w+)?)',
        ]
        for pattern in hold_patterns:
            hold_match = re.search(pattern, s)
            if hold_match:
                holding_object = hold_match.group(1)
                break
    
    # Calculate success probability and progress
    success_prob = 0.0
    progress = 0.0
    
    if has_goal:
        progress = 0.05
        
        # Check if holding target object
        holding_target = False
        if target_object and is_holding and holding_object:
            # More flexible matching
            target_words = set(target_object.lower().split())
            holding_words = set(holding_object.lower().split())
            if target_words & holding_words:  # Any word overlap
                holding_target = True
        
        # Check if at target location
        at_target_location = False
        if target_location:
            loc_words = set(target_location.lower().split())
            state_words = set(s.split())
            if loc_words & state_words:  # Any word overlap
                at_target_location = True
        
        # Critical: holding target object at target location = near-success
        if holding_target and at_target_location:
            success_prob = 0.95
            progress = 0.35
        # Holding target object (but not at location)
        elif holding_target:
            success_prob = 0.70
            progress = 0.25
        # Target object visible but not held
        elif target_object and target_object.lower() in s:
            success_prob = 0.45
            progress = 0.15
        # Target object not yet found
        else:
            success_prob = 0.15
            progress = 0.05
        
        # At target location bonus (if not already accounted for)
        if at_target_location and not (holding_target and at_target_location):
            success_prob = min(0.95, success_prob + 0.15)
            progress = min(0.40, progress + 0.15)
    
    # Object state prerequisites
    if "is clean" in s:
        success_prob = min(0.95, success_prob + 0.08)
        progress = min(0.40, progress + 0.10)
    if "is plugged in" in s:
        success_prob = min(0.95, success_prob + 0.08)
        progress = min(0.40, progress + 0.10)
    if " is open" in s:
        success_prob = min(0.95, success_prob + 0.05)
        progress = min(0.40, progress + 0.08)
    if " is closed" in s:
        progress = min(0.40, progress + 0.03)
    if " is on" in s:
        success_prob = min(0.95, success_prob + 0.05)
        progress = min(0.40, progress + 0.08)
    
    # Locked container obstacle
    if "is locked" in s:
        success_prob = max(0.0, success_prob - 0.10)
        penalty = max(-0.15, penalty - 0.05)
    
    # Calibration bounds - allow success_prob to reach 1.0 for terminal states
    success_prob = max(0.0, min(1.0, success_prob))
    progress = max(0.0, min(0.40, progress))
    penalty = max(-0.15, min(0.0, penalty))
    
    # Total value
    total = success_prob + progress + penalty
    total = max(0.0, min(1.0, total))
    
    return total, {
        "success_prob": success_prob,
        "progress": progress,
        "penalty": penalty
    }