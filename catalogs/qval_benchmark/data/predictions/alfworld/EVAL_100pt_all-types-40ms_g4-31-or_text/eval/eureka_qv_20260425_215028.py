import re

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value Q(s, a) for an ALFWorld environment task.
    The signal emphasizes the critical path: Target Discovery -> Acquisition -> Navigation to Target Location -> Placement.
    It specifically rewards transitions that move the agent closer to the final goal.
    """
    
    state_low = state.lower()
    next_state_low = next_state.lower()
    action_low = action.lower()
    
    # 1. Terminal Success
    if "task completed" in next_state_low or "successfully" in next_state_low:
        return 1.0, {"completion_reward": 1.0}
    
    # 2. Target and Goal Extraction
    # The task description is usually at the start of the state.
    goal_text = state_low.split('\n')[0]
    target_obj = None
    target_loc = None
    is_cleaning_task = False
    
    # Patterns for "put the X in the Y" or "clean the X"
    put_match = re.search(r"put (?:a|the) (.+?) in (?:the|a) ([^.\n]+)", goal_text)
    if put_match:
        target_obj = put_match.group(1).strip().strip('.')
        target_loc = put_match.group(2).strip().strip('.')
    else:
        clean_match = re.search(r"clean (?:the|a) ([^.\n]+)", goal_text)
        if clean_match:
            target_obj = clean_match.group(1).strip().strip('.')
            is_cleaning_task = True

    # 3. Possession and Location Detection
    def check_holding_target(text, obj):
        if not obj: return False
        obj_clean = obj.replace("the ", "").replace("a ", "").strip()
        # Standard ALFWorld "holding" patterns
        if f"holding {obj_clean}" in text or f"carrying {obj_clean}" in text:
            return True
        # Fallback for simpler state descriptions
        if "holding" in text and obj_clean in text:
            return True
        return False

    was_holding_target = check_holding_target(state_low, target_obj)
    is_holding_target = check_holding_target(next_state_low, target_obj)
    
    was_holding_any = any(kw in state_low for kw in ["holding", "carrying"])
    is_holding_any = any(kw in next_state_low for kw in ["holding", "carrying"])

    # 4. Action Categorization
    acquisition_keywords = ["take", "pick up", "grab"]
    placement_keywords = ["put", "place", "drop", "clean"]
    interaction_keywords = ["open", "examine", "look", "search", "close", "inventory", "unlock"]
    movement_keywords = ["go to", "walk to", "move to", "go ", "walk ", "move "]
    
    is_acquisition_action = any(kw in action_low for kw in acquisition_keywords)
    is_placement_action = any(kw in action_low for kw in placement_keywords)
    is_interaction_action = any(kw in action_low for kw in interaction_keywords)
    is_movement_action = any(kw in action_low for kw in movement_keywords)
    
    # 5. Stagnation and Logic Failure Detection
    stagnated = (
        "nothing happens" in next_state_low or 
        "cannot" in next_state_low or 
        "not possible" in next_state_low or 
        "does not" in next_state_low or 
        "already open" in next_state_low or 
        "already closed" in next_state_low or 
        "is already" in next_state_low or
        "is empty" in next_state_low or
        state_low == next_state_low
    )
    
    # Logical errors: trying to pick up something while already holding something,
    # or trying to put something down without holding anything.
    logic_error = (
        (is_acquisition_action and was_holding_any) or 
        (is_placement_action and not was_holding_any and not is_cleaning_task)
    )
    
    stagnation_penalty = -0.5 if (stagnated or logic_error) else 0.0
    
    # 6. Reward Component Calculation
    acquisition_reward = 0.0
    placement_reward = 0.0
    interaction_reward = 0.0
    navigation_reward = 0.0
    discovery_reward = 0.0
    
    # Navigation: Reward moving toward target location, especially while holding the object
    if is_movement_action and not stagnated:
        if is_holding_target:
            if target_loc and target_loc in next_state_low:
                navigation_reward = 0.8
            else:
                navigation_reward = 0.4
        elif target_loc and target_loc in next_state_low:
            navigation_reward = 0.3
        else:
            navigation_reward = 0.1
            
    # Acquisition: Reward picking up the target object
    if (is_acquisition_action and not stagnated and not logic_error) or (is_holding_target and not was_holding_target):
        if is_holding_target:
            acquisition_reward = 0.7
        else:
            acquisition_reward = 0.2
            
    # Placement/Cleaning: Reward the actual act of putting target in location or cleaning it
    # We trigger this if the agent attempts placement while holding target, or if they transition 
    # from holding the target to no longer holding it in the correct location.
    if not stagnated and not logic_error:
        if is_placement_action:
            if (not is_cleaning_task and was_holding_target):
                # Agent attempted to put the object down while holding it
                if target_loc and target_loc in next_state_low:
                    placement_reward = 0.9
                else:
                    placement_reward = 0.4
            elif (is_cleaning_task and target_obj and target_obj in next_state_low):
                # Agent attempted to clean the target object
                if "clean" in next_state_low or "now clean" in next_state_low:
                    placement_reward = 0.9
                else:
                    placement_reward = 0.4
        
        if (was_holding_target and not is_holding_target):
            if target_loc and target_loc in next_state_low:
                placement_reward = max(placement_reward, 0.9)
    
    # Interaction and Discovery
    if is_interaction_action and not stagnated:
        # Discovery: Finding the target object in a container
        discovery_keywords = ["you see", "contains", "there is a", "is in", "found", "reveals"]
        if any(kw in next_state_low for kw in discovery_keywords):
            discovery_reward = 0.6 if (target_obj and target_obj in next_state_low) else 0.2
        
        # General interaction: Reward opening containers or searching
        is_informational = any(kw in action_low for kw in ["inventory", "look"])
        if not is_informational:
            if is_holding_target:
                interaction_reward = 0.2
            elif target_obj and target_obj in next_state_low:
                interaction_reward = 0.4
            elif any(kw in next_state_low for kw in ["open", "unlock"]):
                interaction_reward = 0.2
            
    step_penalty = -0.05
    
    total = (
        acquisition_reward + 
        placement_reward + 
        interaction_reward + 
        navigation_reward + 
        discovery_reward + 
        stagnation_penalty + 
        step_penalty
    )
    
    return total, {
        "acquisition_reward": acquisition_reward,
        "placement_reward": placement_reward,
        "interaction_reward": interaction_reward,
        "navigation_reward": navigation_reward,
        "discovery_reward": discovery_reward,
        "stagnation_penalty": stagnation_penalty,
        "step_penalty": step_penalty,
    }