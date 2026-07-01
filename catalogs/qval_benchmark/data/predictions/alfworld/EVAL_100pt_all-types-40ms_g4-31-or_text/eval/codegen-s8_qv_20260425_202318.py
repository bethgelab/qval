import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) based on the transition from state s to next_state s'.
    The estimate is based on progress milestones toward the goal described in the state.
    """
    # 1. Extract the goal from the state representation.
    # Goals in ALFWorld typically follow the pattern: "Put a [clean/dirty] [object] in the [destination]."
    goal_match = re.search(r"Put (?:a )?(clean|dirty)?\s?(.+?) in the (.+?)(?:\.|\s|$)", state, re.IGNORECASE)
    
    target_obj = None
    dest_obj = None
    cond_obj = None
    
    if goal_match:
        groups = goal_match.groups()
        if len(groups) == 3:
            cond_obj, target_obj, dest_obj = groups
        elif len(groups) == 2:
            target_obj, dest_obj = groups
    
    # Normalize target and destination keywords for flexible matching
    t_obj = target_obj.lower().strip() if target_obj else None
    d_obj = dest_obj.lower().strip() if dest_obj else None
    c_obj = cond_obj.lower().strip() if cond_obj else None
    
    # 2. Check for terminal success.
    # Terminal success states provide the maximum reward.
    ns_lower = next_state.lower()
    if "task completed" in ns_lower or "successfully" in ns_lower:
        return 1.0
        
    # 3. Check for failures or invalid actions.
    # Actions that result in an error or an invalid attempt are penalized.
    failure_phrases = ["i cannot", "not found", "invalid", "unable to", "nothing to", "not possible"]
    if any(phrase in ns_lower for phrase in failure_phrases):
        return 0.0
        
    # 4. Check for stagnation.
    # If the state doesn't change or the action had no effect, the Q-value is low.
    if state == next_state:
        return 0.05
        
    # 5. Estimate progress milestones in next_state.
    # We map the common sequence of steps in ALFWorld to a value between 0.1 and 0.9.
    # Sequence: Find Object -> Pick Up -> Clean (if needed) -> Navigate to Destination -> Place.
    progress = 0.1  # Base value for any non-failing action
    
    if t_obj:
        # Milestone: Target object is mentioned in the observation (found it)
        if t_obj in ns_lower:
            progress = 0.2
            
            # Milestone: Agent is holding the target object
            if f"holding {t_obj}" in ns_lower or f"picked up {t_obj}" in ns_lower:
                progress = 0.4
                
                # Milestone: Object condition met (e.g., it's now clean)
                if c_obj and (f"{t_obj} is {c_obj}" in ns_lower or f"cleaned {t_obj}" in ns_lower):
                    progress = 0.6
                
                # Milestone: Agent is at the destination while holding the target
                if d_obj and d_obj in ns_lower:
                    progress = 0.8
    
    # Milestone: Agent successfully navigated to the destination room, even if not holding the item yet
    if d_obj and d_obj in ns_lower:
        progress = max(progress, 0.3)
        
    return progress