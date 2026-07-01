def signal_function(state: str, action: str, next_state: str):
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    
    # Success indicators
    success_keywords = ['success', 'done', 'completed', 'finished', 'goal', 'succeed', 'successfully', 'task completed', 'task is done', 'you have']
    is_success = any(kw in next_state_lower for kw in success_keywords)
    
    # Error indicators
    error_keywords = ['cannot', 'not', 'already', 'nothing', 'not here', 'nothing happened', 'cannot go', 'not in', 'not a', 'cannot be', 'nothing there', 'already have', 'already on', 'nothing to', 'not possible', 'not available']
    is_error = any(kw in next_state_lower for kw in error_keywords)
    
    # State change detection
    state_changed = state != next_state
    
    # Holding detection
    holding_patterns = ['holding', 'you are holding', 'inventory', 'you take', 'took the', 'picked up', 'grabbed', 'in your inventory', 'holding the', 'have the']
    is_holding = any(pattern in next_state_lower for pattern in holding_patterns)
    was_holding = any(pattern in state_lower for pattern in holding_patterns)
    
    # Action type detection
    is_taking = 'take' in action_lower or 'pick up' in action_lower or 'grab' in action_lower
    is_putting = 'put' in action_lower or 'drop' in action_lower or 'place' in action_lower
    is_navigation = 'go to' in action_lower
    is_manipulation = any(a in action_lower for a in ['take', 'put', 'clean', 'heat', 'cool', 'use', 'open', 'close', 'drop', 'place'])
    is_examination = 'examine' in action_lower
    
    # Stuck detection
    is_stuck = state_lower == next_state_lower and not is_success and not is_error
    
    # Goal objects
    common_goal_objects = ['bottle', 'mug', 'cup', 'spoon', 'fork', 'knife', 'bowl', 'plate', 'book', 'phone', 'key', 'watch', 'soap', 'sponge', 'towel', 'cloth', 'batteries', 'lightbulb']
    goal_object_patterns = [obj for obj in common_goal_objects if obj in state_lower or obj in next_state_lower]
    is_goal_object_action = is_taking and len(goal_object_patterns) > 0
    is_progress_navigation = is_navigation and is_holding
    
    # Episode progress estimation (0-1 scale, 1 = late in episode)
    object_count = sum(1 for obj in common_goal_objects if obj in state_lower)
    room_keywords = ['kitchen', 'bedroom', 'bathroom', 'living room', 'dining room', 'closet', 'counter', 'table', 'shelf', 'drawer', 'sink', 'stove', 'fridge', 'microwave', 'cabinet']
    room_count = sum(1 for room in room_keywords if room in state_lower)
    episode_progress = min(1.0, (object_count * 0.1 + room_count * 0.05))
    
    # Temporal discount - earlier actions should have higher value
    temporal_discount = 1.0 - (episode_progress * 0.35)
    
    # Base signal
    base_signal = 0.0
    if is_success:
        base_signal = 1.0 * temporal_discount
    elif is_error:
        base_signal = -0.6
    elif is_stuck:
        base_signal = -0.7
    elif is_manipulation and state_changed:
        base_signal = 0.4 if is_goal_object_action else 0.3
    elif is_manipulation and not state_changed:
        base_signal = -0.25
    elif is_navigation and state_changed:
        base_signal = 0.3 if is_progress_navigation else 0.2
    elif is_navigation and not state_changed:
        base_signal = -0.3
    elif is_examination and state_changed:
        base_signal = 0.1
    else:
        base_signal = -0.2
    
    # Apply temporal discount to positive base signals (except success which already has it)
    if base_signal > 0 and not is_success:
        base_signal = base_signal * temporal_discount
    
    # Action bonus
    action_bonus = 0.0
    if state_changed and not is_error and not is_stuck:
        if 'take' in action_lower:
            action_bonus = 0.3 * temporal_discount
        elif 'put' in action_lower or 'drop' in action_lower or 'place' in action_lower:
            action_bonus = 0.25 * temporal_discount
        elif 'clean' in action_lower:
            action_bonus = 0.3 * temporal_discount
        elif 'open' in action_lower or 'close' in action_lower:
            action_bonus = 0.2 * temporal_discount
        elif 'heat' in action_lower or 'cool' in action_lower:
            action_bonus = 0.25 * temporal_discount
        elif 'use' in action_lower:
            action_bonus = 0.2 * temporal_discount
    
    # Redundancy penalty
    redundancy_penalty = 0.0
    if not state_changed and not is_success:
        if is_navigation:
            redundancy_penalty = -0.35
        elif is_manipulation:
            redundancy_penalty = -0.3
        elif is_examination:
            redundancy_penalty = -0.2
        else:
            redundancy_penalty = -0.25
    
    # Wrong object penalty
    wrong_object_penalty = 0.0
    if is_holding and not is_success:
        if not state_changed or is_error or is_stuck:
            if is_taking:
                wrong_object_penalty = -0.35
            elif is_navigation:
                wrong_object_penalty = -0.3
            else:
                wrong_object_penalty = -0.25
    
    # Goal progress bonus - more conservative, only for clear progress
    goal_progress_bonus = 0.0
    
    # Clear goal object retrieval
    if is_goal_object_action and state_changed and not is_error:
        goal_progress_bonus = 0.25 * temporal_discount
    
    # Navigation while holding (clearly on task)
    if is_progress_navigation and state_changed and not is_error:
        goal_progress_bonus += 0.15 * temporal_discount
    
    # Exploration navigation (minimal bonus)
    if not is_goal_object_action and is_navigation and state_changed and not is_error:
        goal_progress_bonus += 0.03 * temporal_discount
    
    # Goal location movement
    goal_location_keywords = ['kitchen', 'bedroom', 'bathroom', 'living room', 'dining room', 'closet', 'counter', 'table', 'shelf', 'drawer', 'sink', 'stove', 'fridge', 'microwave', 'cabinet']
    moved_to_goal_location = any(loc in next_state_lower for loc in goal_location_keywords) and is_navigation and state_changed
    if moved_to_goal_location and not is_error:
        goal_progress_bonus += 0.08 * temporal_discount
    
    # State change progress (cleaning, heating, cooling)
    state_change_progress = any(kw in next_state_lower for kw in ['clean', 'dirty', 'heated', 'cooled', 'hot', 'cold', 'open', 'closed']) and state_changed
    if state_change_progress and not is_error and not is_stuck:
        goal_progress_bonus += 0.06 * temporal_discount
    
    # Object state progress
    object_state_progress = any(kw in next_state_lower for kw in ['now', 'is now', 'became', 'changed', 'updated']) and state_changed
    if object_state_progress and not is_error and not is_stuck:
        goal_progress_bonus += 0.04 * temporal_discount
    
    # Successful placement
    successful_placement = is_putting and state_changed and not is_error and not is_stuck
    if successful_placement:
        goal_progress_bonus += 0.12 * temporal_discount
    
    # Successful retrieval
    successful_retrieval = is_taking and state_changed and not is_error and not is_stuck
    if successful_retrieval:
        goal_progress_bonus += 0.12 * temporal_discount
    
    # Dropped object
    dropped_object = was_holding and not is_holding and state_changed and not is_error
    if dropped_object:
        goal_progress_bonus += 0.08 * temporal_discount
    
    # Calculate total
    total = base_signal + action_bonus + redundancy_penalty + wrong_object_penalty + goal_progress_bonus
    
    # Clamp
    total = max(-1.0, min(1.0, total))
    
    return total, {
        "base_signal": base_signal,
        "action_bonus": action_bonus,
        "redundancy_penalty": redundancy_penalty,
        "wrong_object_penalty": wrong_object_penalty,
        "goal_progress_bonus": goal_progress_bonus,
        "temporal_discount": temporal_discount,
        "episode_progress": episode_progress,
    }