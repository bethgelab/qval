import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the ALFWorld environment.
    The estimate is based on parsing the goal from the state and checking if the action
    or the resulting next_state shows progress towards fulfilling that goal.
    """
    # 1. Extract the goal from the current state
    goal_match = re.search(r"Goal:\s*(.*?)(?:\.|$)", state, re.IGNORECASE)
    if not goal_match:
        return 0.0
    
    # Clean the goal string for easier parsing
    goal_text = goal_match.group(1).lower().strip().rstrip('.')
    
    def clean_term(t):
        """Removes common articles to normalize object and location names."""
        if not t:
            return ""
        t = t.lower().strip()
        for prefix in ["the ", "a ", "an "]:
            if t.startswith(prefix):
                t = t[len(prefix):]
        return t.strip()

    target_obj = None
    target_loc = None
    target_type = None # 'put', 'clean', 'move'

    # 2. Parse the goal to identify the target object, location, and task type
    if "put" in goal_text or "place" in goal_text:
        target_type = "put"
        # Matches "put [the] apple in [the] fridge"
        m = re.search(r"(?:put|place)\s+(?:the\s+)?(.*?)\s+(?:in|on|at)\s+(?:the\s+)?(.*?)$", goal_text)
        if m:
            target_obj = clean_term(m.group(1))
            target_loc = clean_term(m.group(2))
    elif "clean" in goal_text:
        target_type = "clean"
        # Matches "clean [the] apple"
        m = re.search(r"clean\s+(?:the\s+)?(.*)$", goal_text)
        if m:
            target_obj = clean_term(m.group(1))
    elif "move" in goal_text:
        target_type = "move"
        # Matches "move [the] apple to [the] table"
        m = re.search(r"move\s+(?:the\s+)?(.*?)\s+(?:to|in|on|at)\s+(?:the\s+)?(.*?)$", goal_text)
        if m:
            target_obj = clean_term(m.group(1))
            target_loc = clean_term(m.group(2))

    # If we cannot identify the goal, we cannot reason about the action
    if not target_obj:
        return 0.0

    ns_lower = next_state.lower()
    # Split the next state into sentences/segments for granular analysis
    segments = re.split(r'[.!?]', ns_lower)

    # 3. Evaluate the Next State for Completion (Reward = 1.0)
    if target_type == "put" and target_loc:
        for seg in segments:
            if target_obj in seg and target_loc in seg and any(p in seg for p in ["in", "on", "at"]):
                return 1.0
    elif target_type == "clean":
        for seg in segments:
            if target_obj in seg and "clean" in seg:
                return 1.0
    elif target_type == "move" and target_loc:
        for seg in segments:
            if target_obj in seg and target_loc in seg:
                return 1.0

    # 4. Evaluate the Next State for Progress (Intermediate Rewards)
    for seg in segments:
        # Check if the agent is now holding the required object
        if "holding" in seg and target_obj in seg:
            return 0.6 if target_type in ["put", "move"] else 0.4

    # 5. Evaluate the Action for Utility (Reward based on relevance)
    act_lower = action.lower()
    if target_obj in act_lower:
        if target_type == "put" and ("put" in act_lower or "place" in act_lower):
            if target_loc and target_loc in act_lower:
                return 0.5
            return 0.3
        if target_type == "clean" and "clean" in act_lower:
            return 0.5
        if target_type in ["put", "move"] and "take" in act_lower:
            return 0.4
        if target_type in ["put", "move"] and "open" in act_lower:
            return 0.3
    
    # Handle navigation: if moving to a room mentioned in the action
    if any(move_cmd in act_lower for move_cmd in ["go to", "walk to", "move to"]):
        # Remove the command prefix to isolate the destination name
        dest = act_lower.replace("go to", "").replace("walk to", "").replace("move to", "").strip()
        # If the destination is found in the current 'location' segment of next_state
        for seg in segments:
            if "you are in" in seg and dest in seg:
                return 0.2

    return 0.0