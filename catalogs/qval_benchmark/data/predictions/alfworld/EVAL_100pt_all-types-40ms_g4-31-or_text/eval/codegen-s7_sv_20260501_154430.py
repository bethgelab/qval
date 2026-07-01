import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment.
    The value is based on progress toward the goal:
    1.0: Task success
    0.9: Holding goal object and at target location
    0.7: Holding goal object
    0.4: Goal object found/visible
    0.2: Agent at target location
    0.1: Default starting/searching state
    """
    state_lower = state.lower()
    
    # 1. Success detection: check for explicit completion markers
    if any(phrase in state_lower for phrase in ["task completed", "successfully", "goal achieved"]):
        return 1.0

    # Split state into goal and observation components to avoid false matches
    # ALFWorld states often follow a format like "Goal: ... Observation: ..."
    parts = state.split("Observation:")
    goal_part = parts[0].lower()
    obs_part = parts[1].lower() if len(parts) > 1 else state_lower

    # 2. Goal extraction: find the object and target location
    # Expected pattern: "put a cold apple in the fridge" or "put the book into the cabinet"
    goal_match = re.search(r"put (?:a|the) (.*?) (?:in|into) (?:the|a) (.*?)(?:\.|\n|$)", goal_part)
    if not goal_match:
        # Fallback for less strict pattern
        goal_match = re.search(r"put (.*?) (?:in|into) (.*?)(?:\.|\n|$)", goal_part)

    if goal_match:
        obj = goal_match.group(1).strip().rstrip('.')
        loc = goal_match.group(2).strip().rstrip('.')
        
        # Determine the core noun of the object (e.g., "cold apple" -> "apple")
        obj_words = obj.split()
        main_obj = obj_words[-1] if obj_words else obj

        # 3. Milestone: Holding the object
        # Look for "holding" in the observation part
        if f"holding {obj}" in obs_part or f"holding the {obj}" in obs_part or f"holding {main_obj}" in obs_part:
            # Higher value if the agent is also at the target location
            if loc in obs_part:
                return 0.9
            return 0.7

        # 4. Milestone: Goal object found/visible
        # Check for the full description or just the main noun
        if obj in obs_part or main_obj in obs_part:
            # Ensure we aren't just matching a repetition of the goal text
            return 0.4

        # 5. Milestone: Agent is at/near target location
        if loc in obs_part:
            return 0.2

    # 6. Base value for searching or unknown progress
    return 0.1