import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the ALFWorld environment.
    The Q-value represents the expected discounted cumulative reward, with 1.0 given for task success.
    """
    # 1. Extract goal from the state (usually the beginning of the prompt)
    goal_obj = None
    goal_target = None
    is_cleaning_task = False

    # Common ALFWorld goal patterns
    put_match = re.search(r"put the (.*?) in the (.*?)(?:\.|$)", state, re.IGNORECASE)
    if put_match:
        goal_obj = put_match.group(1).strip().lower()
        goal_target = put_match.group(2).strip().lower()
    else:
        clean_match = re.search(r"clean the (.*?)(?:\.|$)", state, re.IGNORECASE)
        if clean_match:
            goal_obj = clean_match.group(1).strip().lower()
            is_cleaning_task = True

    if not goal_obj:
        # Default if no goal is explicitly parsed
        return 0.1

    # 2. Analyze the next_state for progress milestones
    # We prioritize the most recent observations (end of the text)
    next_state_lower = next_state.lower()
    
    # Terminal success
    success_keywords = ["task completed", "successfully", "placed the", "cleaned the"]
    if any(kw in next_state_lower for kw in success_keywords):
        # Check if the specific object involved is mentioned in the success context
        if goal_obj in next_state_lower:
            return 1.0

    # Milestone: Holding the required object
    holding_obj = False
    if f"holding {goal_obj}" in next_state_lower or f"holding the {goal_obj}" in next_state_lower:
        holding_obj = True

    # Milestone: At the target location
    at_target = False
    if goal_target:
        # Look for presence in the room or seeing the target
        if goal_target in next_state_lower:
            at_target = True
    elif is_cleaning_task:
        if goal_obj in next_state_lower:
            at_target = True

    # 3. Calculate Q-value based on progress milestones
    # The values are approximations of the expected return (gamma^steps * 1.0)
    q_value = 0.0

    if holding_obj and at_target:
        # Almost there: holding object and at destination
        q_value = 0.8
    elif holding_obj:
        # Halfway: object acquired
        q_value = 0.5
    elif at_target:
        # Some progress: reached target (though object not yet held) or reached target object
        q_value = 0.3
    else:
        # Early stage: just navigating or searching
        q_value = 0.1

    # 4. Action-based refinement
    # Boost the value if the action is a constructive move towards the goal
    action_lower = action.lower()
    constructive_actions = ["take", "pick up", "put", "place", "go to", "open"]
    if any(ca in action_lower for ca in constructive_actions):
        # Small boost for active interaction
        q_value += 0.05
    
    # Specifically reward the 'take' action if it leads to holding the object
    if "take" in action_lower and holding_obj:
        q_value += 0.1
    
    # Specifically reward the 'put' or 'clean' action if it leads to success
    if ("put" in action_lower or "clean" in action_lower) and q_value >= 0.8:
        q_value += 0.1

    # Penalty for potentially redundant actions (looking around too much)
    if "look" in action_lower or "examine" in action_lower:
        q_value -= 0.05

    # Clamp and return
    return max(0.0, min(1.0, q_value))