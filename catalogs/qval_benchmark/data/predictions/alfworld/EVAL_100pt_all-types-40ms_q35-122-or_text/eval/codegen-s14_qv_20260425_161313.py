def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Convert to lowercase for easier pattern matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_lower = next_state.lower()
    
    # Base Q-value starts at 0 (no reward expected)
    q_value = 0.0
    
    # Check if task is complete in next_state (immediate reward of 1.0)
    completion_indicators = [
        "task completed", "success", "you have successfully", 
        "congratulations", "completed the task", "you win"
    ]
    if any(ind in next_lower for ind in completion_indicators):
        return 1.0
    
    # Check for failure/terminal states with no success
    failure_indicators = [
        "max steps reached", "step limit exceeded", "failed",
        "could not complete", "task failed"
    ]
    if any(ind in next_lower for ind in failure_indicators):
        return 0.0
    
    # Check if action was valid (not blocked by environment)
    error_patterns = [
        "you can't", "you cannot", "nothing to", "nothing here",
        "nothing to take", "nothing to put", "can't go", "can't move",
        "nothing to", "doesn't exist", "not there", "cannot",
        "cannot be", "cannot move"
    ]
    action_blocked = any(pattern in next_lower for pattern in error_patterns)
    
    if action_blocked:
        q_value -= 0.3  # Significant penalty for invalid action
    
    # Estimate progress toward goal based on action type and state changes
    progress_bonus = 0.0
    
    # Object pickup actions
    if any(word in action_lower for word in ["take", "get", "pick up", "grab"]):
        if "got" in next_lower or "picked up" in next_lower or "you now have" in next_lower:
            progress_bonus += 0.15
    
    # Object placement actions
    if any(word in action_lower for word in ["put", "place", "drop", "on", "in"]):
        if "put" in next_lower or "placed" in next_lower or "on the" in next_lower:
            progress_bonus += 0.15
    
    # Navigation actions
    if any(word in action_lower for word in ["go to", "walk to", "move to", "navigate"]):
        progress_bonus += 0.05
    
    # Object state changes (clean, heat, cool, etc.)
    if any(word in action_lower for word in ["clean", "heat", "cool", "microwave", "wash"]):
        if "clean" in next_lower or "heated" in next_lower or "cooled" in next_lower:
            progress_bonus += 0.1
    
    # Check for inventory state (object held)
    if "you are holding" in next_lower or "you have" in next_lower:
        progress_bonus += 0.1
    
    # Check for proximity to goal location
    goal_keywords = ["receptacle", "table", "counter", "shelf", "sink", "fridge", "microwave"]
    if any(word in next_lower for word in goal_keywords):
        progress_bonus += 0.05
    
    # Penalize backtracking or repetitive actions
    if state_lower == next_lower:
        progress_bonus -= 0.2  # No state change = wasted step
    
    # Discount factor consideration: fewer remaining steps = higher value for progress
    # Estimate steps used by state length (rough proxy)
    state_tokens = len(state.split())
    estimated_steps_used = min(39, max(0, state_tokens // 10))
    steps_remaining = 40 - estimated_steps_used
    
    # More steps remaining = more time to complete, less urgency
    # Fewer steps remaining = higher value for each step if making progress
    if steps_remaining < 10:
        q_value += progress_bonus * 1.5  # Urgency bonus
    else:
        q_value += progress_bonus
    
    # Apply action validity penalty
    if action_blocked:
        q_value -= 0.3
    
    # Clamp to reasonable Q-value range for sparse reward task
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value