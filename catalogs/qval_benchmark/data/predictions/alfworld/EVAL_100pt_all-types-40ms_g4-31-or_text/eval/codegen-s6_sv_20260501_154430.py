import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is based on the agent's progress toward completing the task:
    1.0: Task completed
    0.9: Holding the goal object and at the destination
    0.6: Holding the goal object
    0.4: Goal object is visible in the current state
    0.2: Agent is at the destination (but not holding the object)
    0.0: No significant progress
    """
    state_lower = state.lower()
    
    # 1. Check for terminal success signals
    # ALFWorld typically signals completion with explicit words like 'success'
    success_keywords = ["success", "task completed", "successfully put", "successfully placed"]
    if any(kw in state_lower for kw in success_keywords):
        return 1.0
        
    # 2. Identify the Goal from the state description
    # Expected pattern: "put the [item] in the [target]" or "place the [item] on the [target]"
    goal_pattern = r"(?:put|place|move)\s+the\s+(.*?)\s+(?:in|inside|on|into)\s+the\s+(.*?)(?:\.|\n|$)"
    goal_match = re.search(goal_pattern, state_lower)
    
    if goal_match:
        item = goal_match.group(1).strip()
        target = goal_match.group(2).strip()
        
        # Check for specific completion if not caught by general success keywords
        if f"{item} is now in the {target}" in state_lower or f"{item} is now in {target}" in state_lower:
            return 1.0
            
        # Detect progress milestones
        is_holding = f"holding the {item}" in state_lower or f"holding {item}" in state_lower
        at_target = target in state_lower
        # Item is visible if it's mentioned but not currently held
        item_visible = item in state_lower and not is_holding
        
        if is_holding and at_target:
            return 0.9
        if is_holding:
            return 0.6
        if item_visible:
            return 0.4
        if at_target:
            return 0.2
    else:
        # 3. Fallback heuristic if a concrete goal cannot be parsed from the text
        # If the agent is holding something, it's generally better than holding nothing
        if "holding" in state_lower:
            return 0.5
        # Just in case success is mentioned without a parsed goal
        if "success" in state_lower:
            return 1.0
            
    return 0.0