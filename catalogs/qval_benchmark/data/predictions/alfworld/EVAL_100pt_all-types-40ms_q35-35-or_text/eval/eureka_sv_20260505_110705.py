def signal_function(state: str):
    import re
    
    # Check if episode is already complete
    if "success" in state.lower() or "completed" in state.lower() or "task complete" in state.lower():
        return 1.0, {
            "completion_bonus": 1.0,
        }
    
    state_lower = state.lower()
    
    # Initialize components
    progress_score = 0.0
    step_penalty = 0.0
    task_specific_bonus = 0.0
    location_bonus = 0.0
    goal_proximity = 0.0
    holding_bonus = 0.0
    object_discovered = 0.0
    action_progress = 0.0
    near_completion_bonus = 0.0
    action_sequence_bonus = 0.0
    error_penalty = 0.0
    
    # 1. Step-based penalty (scale with 40-step limit) - more aggressive
    step_match = re.search(r"step\s*(\d+)", state_lower)
    if step_match:
        step_num = int(step_match.group(1))
        remaining_steps = 40 - step_num
        if remaining_steps > 0:
            # Higher penalty as we run out of steps - calibrated for 40-step limit
            step_penalty -= (1 - remaining_steps / 40) * 0.50
    else:
        # No step info means we're early in the episode, but still penalize
        step_penalty -= 0.05
    
    # 2. Error/penalty detection (penalize incorrect actions) - more reliable patterns
    error_patterns = [
        r"nothing to do", r"nothing happens", r"nothing changes",
        r"can't", r"cannot", r"error", r"fail", r"failed",
        r"wrong object", r"incorrect", r"invalid",
        r"empty", r"empty hands", r"no object",
        r"already have", r"already holding", r"already on", r"already in",
        r"already opened", r"already closed",
        r"nothing seems", r"nothing appears",
        r"unusable", r"impossible",
    ]
    error_count = sum(1 for pattern in error_patterns if re.search(pattern, state_lower))
    if error_count > 0:
        error_penalty = min(error_count * 0.15, 0.45)
        step_penalty -= error_penalty
    
    # 3. Object manipulation progress - holding indicates active progress
    holding_patterns = [
        r"you are holding", r"you now hold", r"you're holding",
        r"picked up", r"picked\s+\w+", r"pick up", r"pick\s+\w+",
    ]
    holding_found = sum(1 for pattern in holding_patterns if re.search(pattern, state_lower))
    if holding_found > 0:
        holding_bonus = min(holding_found * 0.18, 0.40)
        progress_score += holding_bonus
    
    # 4. Object discovery - detect when agent has found relevant objects
    object_patterns = [
        r"you see", r"there is", r"on the", r"on a",
        r"on the counter", r"on the table", r"on the desk",
        r"on the sink", r"on the stove", r"in the", r"in a",
        r"on the floor", r"on the ground",
    ]
    object_found = sum(1 for pattern in object_patterns if re.search(pattern, state_lower))
    if object_found > 0:
        object_discovered = 0.08
        progress_score += object_discovered
    
    # 5. Location progress - being at relevant locations (reduced baseline)
    location_patterns = [
        "kitchen", "bedroom", "bathroom", "living room", "dining room",
        "countertop", "desk", "table", "cabinet", "drawer", "sink",
        "stove", "microwave", "toilet", "bathtub", "shower", "counter",
        "floor", "sidetable", "nightstand",
    ]
    found_locations = sum(1 for loc in location_patterns if loc in state_lower)
    location_bonus = min(found_locations * 0.04, 0.12)
    progress_score += location_bonus
    
    # 6. Action completion hints
    action_completions = [
        "successfully", "done", "complete", "finished", "placed", "moved",
        "opened", "closed", "washed", "cleaned", "turned on", "turned off",
        "put", "put on", "put in", "put under", "taken", "taken from",
        "filled", "emptied", "heated", "cooled", "cooked", "eaten",
        "dried", "wiped", "scrubbed", "rinsed", "soaked",
    ]
    found_completions = sum(1 for action in action_completions if action in state_lower)
    action_progress = min(found_completions * 0.08, 0.24)
    progress_score += action_progress
    
    # 7. Task-specific sub-goal detection
    task_indicators = {
        "put_on": [r"on\s+\w+", "placed on", "put on"],
        "in_container": [r"in\s+\w+", "inside", "put in", "put into", "into"],
        "cleaned": ["clean", "cleaned", "washed", "wipe", "wiped", "scrub"],
        "cooked": ["cook", "cooked", "heated", "microwave", "stove", "oven"],
        "opened": ["open", "opened", "unlocked"],
        "closed": ["close", "closed", "shut"],
        "filled": ["fill", "filled", "water", "liquid"],
        "emptied": ["empty", "emptied", "trash", "garbage"],
        "dried": ["dry", "dried", "towel", "paper towel"],
        "moved_to": ["move", "moved", "transport", "bring"],
    }
    for task_type, patterns in task_indicators.items():
        task_found = sum(1 for pattern in patterns if re.search(pattern, state_lower))
        if task_found > 0:
            task_specific_bonus += min(task_found * 0.08, 0.32)
    
    # 8. Goal-related keywords
    goal_keywords = [
        "goal", "target", "task", "mission", "objective", "required", "need",
        "place", "move", "bring", "get", "find", "locate",
    ]
    goal_count = sum(1 for kw in goal_keywords if kw in state_lower)
    goal_proximity = min(goal_count * 0.04, 0.16)
    progress_score += goal_proximity
    
    # 9. Action sequence bonus - coherent multi-step planning
    action_sequence_patterns = [
        (r"pick up", r"hold"),
        (r"open", r"inside"),
        (r"place", r"on"),
        (r"put", r"in"),
        (r"move", r"to"),
        (r"bring", r"to"),
        (r"wash", r"clean"),
    ]
    for pattern1, pattern2 in action_sequence_patterns:
        if re.search(pattern1, state_lower) and re.search(pattern2, state_lower):
            action_sequence_bonus += 0.10
    action_sequence_bonus = min(action_sequence_bonus, 0.30)
    progress_score += action_sequence_bonus
    
    # 10. Goal completion patterns
    goal_completion_patterns = [
        r"place.*on", r"put.*in", r"put.*on", r"move.*to", r"bring.*to",
        r"place.*in", r"put.*on", r"put.*in",
    ]
    for pattern in goal_completion_patterns:
        if re.search(pattern, state_lower):
            goal_proximity += 0.06
    goal_proximity = min(goal_proximity, 0.22)
    progress_score += goal_proximity
    
    # 11. Near completion bonus - detect when multiple sub-goals are satisfied
    near_completion_patterns = [
        "holding", "placed", "put", "moved", "opened", "closed", "washed",
        "filled", "emptied", "dried", "cooked", "heated", "turned on",
    ]
    near_completion_found = sum(1 for pattern in near_completion_patterns if pattern in state_lower)
    if near_completion_found >= 3:
        near_completion_bonus = 0.30
    elif near_completion_found >= 2:
        near_completion_bonus = 0.18
    elif near_completion_found >= 1:
        near_completion_bonus = 0.10
    progress_score += near_completion_bonus
    
    # Calculate total with bounds
    total = progress_score + task_specific_bonus + step_penalty + near_completion_bonus
    total = max(0.05, min(1.0, total))
    
    return total, {
        "progress_score": progress_score,
        "task_specific_bonus": task_specific_bonus,
        "step_penalty": step_penalty,
        "location_bonus": location_bonus,
        "goal_proximity": goal_proximity,
        "holding_bonus": holding_bonus,
        "object_discovered": object_discovered,
        "action_progress": action_progress,
        "near_completion_bonus": near_completion_bonus,
        "action_sequence_bonus": action_sequence_bonus,
        "error_penalty": error_penalty,
    }