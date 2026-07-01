def signal_function(state: str, action: str, next_state: str):
    import re
    
    progress_reward = 0.0
    safety_penalty = 0.0
    step_efficiency = 0.0
    transition_quality = 0.0
    goal_proximity = 0.0
    total = 0.0
    
    completion_indicators = ['done', 'complete', 'finished', 'success', 'goal', 'reached', 'achieved', 'task complete', 'task done', 'completed']
    progress_indicators = ['pick', 'place', 'put', 'take', 'bring', 'move', 'clean', 'open', 'close', 'wash', 'dry', 'pick up', 'place on', 'put in', 'take from', 'grab', 'drop', 'inspect']
    error_indicators = ['error', 'invalid', 'cannot', 'not allowed', 'failed', 'impossible', 'already', 'nothing', 'empty', 'out of', 'unreachable', 'does not exist']
    success_indicators = ['success', 'successful', 'completed', 'done', 'finished', 'achieved', 'reached', 'task complete']
    
    state_lower = state.lower()
    next_lower = next_state.lower()
    action_lower = action.lower()
    
    state_has_completion = any(word in state_lower for word in completion_indicators)
    next_has_completion = any(word in next_lower for word in completion_indicators)
    state_has_progress = any(word in state_lower for word in progress_indicators)
    next_has_progress = any(word in next_lower for word in progress_indicators)
    next_has_success = any(word in next_lower for word in success_indicators)
    next_has_error = any(word in next_lower for word in error_indicators)
    
    is_navigation = any(word in action_lower for word in ['go', 'move', 'navigate', 'walk', 'head', 'travel', 'path', 'to', 'walk to', 'go to'])
    is_manipulation = any(word in action_lower for word in ['pick', 'place', 'put', 'take', 'bring', 'clean', 'open', 'close', 'wash', 'dry', 'grab', 'drop', 'inspect', 'examine'])
    is_communication = any(word in action_lower for word in ['say', 'tell', 'ask', 'request', 'command', 'speak'])
    
    # Progress reward - focused on actual goal advancement
    if next_has_completion and not state_has_completion:
        progress_reward = 0.85
    elif next_has_success and not state_has_progress:
        progress_reward = 0.72
    elif next_has_progress and not state_has_completion:
        progress_reward = 0.55
    elif state_has_completion:
        progress_reward = 0.65
    elif state_has_progress:
        progress_reward = 0.42
    else:
        progress_reward = 0.25
    
    # Safety penalty - clear errors get heavy penalty
    if next_has_error:
        safety_penalty = -0.55
    elif 'invalid' in action_lower or 'cannot' in action_lower:
        safety_penalty = -0.38
    elif is_navigation and not next_has_progress:
        safety_penalty = -0.15
    else:
        safety_penalty = -0.08
    
    # Step efficiency - reward actions that change state meaningfully
    state_changed = state != next_state
    if state_changed:
        # Check if action type matches expected behavior for state change
        if is_manipulation and next_has_progress:
            step_efficiency = 0.35
        elif is_manipulation:
            step_efficiency = 0.25
        elif is_navigation and next_has_progress:
            step_efficiency = 0.28
        elif is_navigation:
            step_efficiency = 0.15
        elif is_communication and next_has_progress:
            step_efficiency = 0.22
        else:
            step_efficiency = 0.12
    else:
        # No state change - significant penalty for wasted action
        step_efficiency = -0.45
    
    # Transition quality - assess information gain from state change
    if state_changed and not next_has_error:
        # More progress words in next state = better transition
        progress_count = sum(1 for w in progress_indicators if w in next_lower)
        completion_count = sum(1 for w in completion_indicators if w in next_lower)
        if completion_count > 0:
            transition_quality = 0.35
        elif progress_count > 0:
            transition_quality = 0.22
        else:
            transition_quality = 0.12
    elif state_changed:
        transition_quality = 0.05
    else:
        transition_quality = -0.35
    
    # Goal proximity - detect if we're getting closer to task completion
    # Check for task-specific keywords in state
    task_keywords = ['table', 'chair', 'sofa', 'bed', 'shelf', 'countertop', 'floor', 'desk', 'sink', 'bathroom', 'kitchen', 'living room', 'bedroom', 'hallway', 'door', 'window']
    state_task_keywords = sum(1 for kw in task_keywords if kw in state_lower)
    next_task_keywords = sum(1 for kw in task_keywords if kw in next_lower)
    
    # Check for object interaction keywords
    object_keywords = ['cup', 'bottle', 'book', 'lamp', 'phone', 'remote', 'keys', 'medicine', 'soap', 'towel', 'blanket', 'pillow', 'plate', 'utensil', 'food', 'toy', 'plant', 'picture', 'clock', 'mirror', 'rug', 'mat', 'box', 'bag', 'suitcase', 'backpack', 'glasses', 'shoes', 'hat', 'jacket', 'coat', 'belt', 'watch', 'ring', 'necklace', 'bracelet', 'earrings', 'pen', 'pencil', 'notebook', 'paper', 'folder', 'file', 'document', 'screen', 'monitor', 'keyboard', 'mouse', 'charger', 'cable', 'wire', 'battery', 'power', 'light', 'bulb', 'switch', 'button', 'lever', 'handle', 'knob', 'lock', 'key', 'code', 'password', 'pin', 'number', 'digit', 'symbol', 'sign', 'label', 'tag', 'mark', 'sticker', 'tape', 'glue', 'paint', 'brush', 'roller', 'sponge', 'cloth', 'rag', 'paper towel', 'napkin', 'tissue', 'handkerchief', 'mask', 'glove', 'apron', 'hat', 'cap', 'hoodie', 'sweater', 'shirt', 't-shirt', 'dress', 'skirt', 'pants', 'jeans', 'shorts', 'socks', 'underwear', 'panties', 'bra', 'panties', 'leggings', 'boots', 'sandals', 'slippers', 'heels', 'sneakers', 'running shoes', 'trainers', 'loafers', 'oxfords', 'moccasins', 'clogs', 'flip flops', 'platforms', 'wedges', 'heels']
    state_object_count = sum(1 for obj in object_keywords if obj in state_lower)
    next_object_count = sum(1 for obj in object_keywords if obj in next_lower)
    
    # More objects interacted with = closer to task completion
    if next_object_count > state_object_count and not next_has_completion:
        goal_proximity = 0.28
    elif next_object_count < state_object_count:
        goal_proximity = -0.15
    elif state_has_progress and not next_has_progress:
        goal_proximity = 0.18
    elif state_has_completion:
        goal_proximity = 0.25
    else:
        goal_proximity = 0.08
    
    total = progress_reward + safety_penalty + step_efficiency + transition_quality + goal_proximity
    
    return total, {
        "progress_reward": progress_reward,
        "safety_penalty": safety_penalty,
        "step_efficiency": step_efficiency,
        "transition_quality": transition_quality,
        "goal_proximity": goal_proximity,
    }