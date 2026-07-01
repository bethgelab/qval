import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Normalize strings for case-insensitive matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # 1. Check for immediate success in the next state
    success_keywords = ["success", "complete", "done", "task complete", "success!"]
    for kw in success_keywords:
        if kw in next_state_lower:
            return 1.0
            
    # 2. Extract task goal from the current state
    # ALFWorld tasks typically follow "Task: ..." or are implied by the context.
    task_match = re.search(r"task:\s*(.+)", state_lower)
    task_intent = task_match.group(1) if task_match else ""
    
    score = 0.1  # Baseline value for continuing
    
    # 3. Check for specific task completion conditions
    # Pattern: "put the [obj] in the [loc]"
    put_match = re.search(r"put\s+the\s+(\w+)\s+in\s+the\s+(\w+)", task_intent)
    if put_match:
        obj, loc = put_match.groups()
        # Check if next_state shows the object is now in the target location
        # Pattern: "[obj] is in the [loc]" or "[obj] in the [loc]"
        goal_pattern = rf"{obj}\s+(in|on)\s+the\s+{loc}"
        if re.search(goal_pattern, next_state_lower):
            score = max(score, 0.95)
        elif re.search(goal_pattern, state_lower):
            # Already satisfied in current state
            score = max(score, 0.9)
            
    # 4. Check for cleaning tasks
    clean_match = re.search(r"(clean|wash)\s+the\s+(\w+)", task_intent)
    if clean_match:
        verb, obj = clean_match.groups()
        # If action matches cleaning and next state confirms
        if re.search(rf"{verb}\s+the\s+{obj}", action_lower):
            if re.search(rf"{verb}\s+the\s+{obj}", next_state_lower):
                score = max(score, 0.8)
            elif re.search(rf"{obj}\s+is\s+clean", next_state_lower):
                score = max(score, 0.85)
                
    # 5. Check for navigation progress
    # If action is "go to [room]" and next state shows agent in that room
    if "go to" in action_lower:
        # Extract room name (simplified)
        room_match = re.search(r"go\s+to\s+(the\s+)?(\w+)", action_lower)
        if room_match:
            room = room_match.group(2)
            # Check if next state indicates being in that room
            if re.search(rf"in\s+the\s+{room}", next_state_lower):
                score = max(score, 0.6)
                
    # 6. Check for taking objects
    if "take" in action_lower:
        if "in your hand" in next_state_lower or "holding" in next_state_lower:
            score = max(score, 0.6)
        # If object is no longer in its original location (simplified)
        elif "on the table" not in next_state_lower and "on the table" in state_lower:
            score = max(score, 0.5)
            
    # 7. Check for putting objects (without explicit task match)
    if "put" in action_lower:
        # If next state shows object is in a container/room
        if "in the" in next_state_lower or "on the" in next_state_lower:
            score = max(score, 0.5)
            
    # 8. Efficiency Bonus: If next_state looks more "resolved" than state
    # Count "not found" or "not in" occurrences as negative indicators
    state_negatives = state_lower.count("not") + state_lower.count("fail")
    next_state_negatives = next_state_lower.count("not") + next_state_lower.count("fail")
    if next_state_negatives < state_negatives:
        score = max(score, 0.3)
        
    # Ensure score is within valid Q-value range [0, 1]
    return min(1.0, max(0.0, score))