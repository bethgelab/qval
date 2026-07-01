import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for task completion - highest Q-value
    completion_patterns = [
        "task is complete", "success", "done", "completed", 
        "goal achieved", "task completed", "successfully"
    ]
    for pattern in completion_patterns:
        if pattern in next_state.lower():
            return 1.0
    
    # Check for failure/invalid action - lowest Q-value
    failure_patterns = [
        "nothing", "cannot", "already", "invalid", "impossible",
        "not found", "not available", "does not exist", "no such"
    ]
    for pattern in failure_patterns:
        if pattern in next_state.lower():
            return 0.05
    
    # Estimate progress score (0 to 0.7)
    progress_score = 0.0
    
    # Check if state changed meaningfully
    if next_state != state:
        # Look for positive change indicators
        if "now" in next_state.lower():
            progress_score += 0.25
        if "is now" in next_state.lower():
            progress_score += 0.2
        
        # Check for location/object state changes
        if "on the" in next_state.lower() or "in the" in next_state.lower():
            progress_score += 0.15
        if "clean" in next_state.lower() or "turned on" in next_state.lower():
            progress_score += 0.15
    
    # Evaluate action type effectiveness
    action_lower = action.lower()
    if "put" in action_lower or "place" in action_lower:
        progress_score += 0.25
    elif "take" in action_lower or "pickup" in action_lower:
        progress_score += 0.2
    elif "go" in action_lower or "navigate" in action_lower:
        progress_score += 0.1
    elif "clean" in action_lower:
        progress_score += 0.15
    elif "turn on" in action_lower or "turn off" in action_lower:
        progress_score += 0.15
    elif "open" in action_lower or "close" in action_lower:
        progress_score += 0.1
    
    # Check for goal-related progress in state
    goal_keywords = ["goal", "target", "destination", "place", "put", "move"]
    for keyword in goal_keywords:
        if keyword in next_state.lower():
            progress_score += 0.05
    
    # Cap progress score
    progress_score = min(progress_score, 0.7)
    
    # Add base exploration value
    base_value = 0.1
    
    return base_value + progress_score