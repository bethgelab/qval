def signal_function(state: str, action: str, next_state: str):
    """
    Estimate Q-value for ALFWorld tasks with strict object identity checking
    and better calibration to avoid overestimating irrelevant actions.
    """
    # Reward components
    goal_progress = 0.0
    action_validity = 0.0
    efficiency = 0.0
    location_relevance = 0.0
    state_change_quality = 0.0
    task_proximity = 0.0
    object_relevance = 0.0
    object_mismatch_penalty = 0.0
    
    # Normalize strings for analysis
    state_lower = state.lower()
    action_lower = action.lower()
    next_lower = next_state.lower()
    
    import re
    
    # Extract target object from goal description
    target_object = None
    patterns = [
        r'put\s+the\s+([a-z]+)',
        r'put\s+([a-z]+)\s+on',
        r'put\s+([a-z]+)\s+in',
        r'find\s+the\s+([a-z]+)',
        r'take\s+the\s+([a-z]+)',
        r'move\s+the\s+([a-z]+)',
        r'bring\s+the\s+([a-z]+)',
        r'pick\s+up\s+the\s+([a-z]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, state_lower)
        if match:
            target_object = match.group(1)
            break
    
    # Also check for "the [object]" patterns in general context
    if target_object is None:
        the_patterns = [r'the\s+([a-z]+(?:\s+[a-z]+)?)']
        for pattern in the_patterns:
            matches = re.findall(pattern, state_lower)
            for m in matches:
                # Filter out common non-object words
                if m not in ['task', 'completed', 'state', 'action', 'current', 'next']:
                    target_object = m.strip()
                    break
            if target_object:
                break
    
    # Validate action command format
    valid_prefixes = [
        'go to', 'take', 'put', 'open', 'close', 'clean', 
        'heat', 'cool', 'toggle', 'look', 'inventory', 'examine'
    ]
    is_valid = any(action_lower.startswith(p) for p in valid_prefixes)
    
    # Identify failure conditions
    is_failure = False
    failure_phrases = [
        'nothing happens', 'cannot', "can't", 'not available', 'not visible',
        'you do not have', 'you are not holding', 'already', 'no thing', 'no object',
        'there is no', 'you cannot', 'you are not at'
    ]
    for phrase in failure_phrases:
        if phrase in next_lower:
            is_failure = True
            break
    
    # Detect unproductive search patterns
    is_unproductive_search = False
    if 'go to' in action_lower:
        if any(phrase in next_lower for phrase in ['nothing', 'empty', 'no', 'does not exist']):
            is_unproductive_search = True
    
    # Check for action loops
    is_loop = False
    if 'you are at' in state_lower and 'you are at' in next_lower:
        state_loc = state_lower.split('you are at')[-1].split('\n')[0][:30]
        next_loc = next_lower.split('you are at')[-1].split('\n')[0][:30]
        if state_loc == next_loc and 'go to' in action_lower:
            is_loop = True
    
    # Detect if we're holding something
    holding_something = 'holding' in next_lower or 'you have' in next_lower
    
    # Check if holding the TARGET object specifically
    is_holding_target = False
    if target_object and holding_something:
        holding_context_start = next_lower.find('holding')
        if holding_context_start != -1:
            holding_context = next_lower[holding_context_start:holding_context_start+150]
            if target_object in holding_context.lower():
                is_holding_target = True
    
    # Check if holding WRONG object (not target)
    is_holding_wrong_object = False
    if holding_something and target_object:
        holding_context_start = next_lower.find('holding')
        if holding_context_start != -1:
            holding_context = next_lower[holding_context_start:holding_context_start+150]
            # Extract what we're holding
            holding_match = re.search(r'holding\s+the\s+([a-z]+)', holding_context)
            if holding_match:
                held_object = holding_match.group(1)
                if held_object != target_object:
                    is_holding_wrong_object = True
    
    # Check for task completion
    is_task_complete = False
    if any(phrase in next_lower for phrase in ['task completed', 'completed', 'success', 'task is complete']):
        is_task_complete = True
    
    # Validate action with target object alignment
    action_matches_target = False
    if target_object:
        if target_object in action_lower:
            action_matches_target = True
        elif 'take' in action_lower or 'put' in action_lower:
            # Check if action involves any object - if so, it should match target
            object_patterns = [r'take\s+the\s+([a-z]+)', r'put\s+the\s+([a-z]+)', r'put\s+([a-z]+)\s+on']
            for pattern in object_patterns:
                match = re.search(pattern, action_lower)
                if match:
                    action_object = match.group(1)
                    if action_object != target_object:
                        # Wrong object in action
                        action_matches_target = False
                        break
                    else:
                        action_matches_target = True
                        break
    
    # Evaluate action validity with stricter penalties for object mismatch
    if is_valid:
        if action_lower.startswith('go to'):
            action_validity = 0.05
        elif action_lower.startswith('take'):
            if target_object and action_matches_target:
                action_validity = 0.35
            elif target_object and not action_matches_target:
                action_validity = -0.20  # Taking wrong object
            else:
                action_validity = 0.15
        elif action_lower.startswith('put'):
            if target_object and action_matches_target:
                action_validity = 0.50
            elif target_object and not action_matches_target:
                action_validity = -0.30  # Putting wrong object
            else:
                action_validity = 0.25
        elif action_lower.startswith('open'):
            action_validity = 0.12
        elif action_lower.startswith('close'):
            action_validity = 0.04
        elif action_lower.startswith('clean') or action_lower.startswith('heat') or action_lower.startswith('cool'):
            if target_object and action_matches_target:
                action_validity = 0.25
            elif target_object and not action_matches_target:
                action_validity = -0.15
            else:
                action_validity = 0.10
        elif action_lower.startswith('look') or action_lower.startswith('examine'):
            action_validity = 0.01
        else:
            action_validity = 0.02
        
        if is_failure:
            action_validity -= 0.40
    else:
        action_validity = -0.45
    
    # Assess location relevance - more conservative values
    location_relevance = 0.0
    high_value_locs = ['sink', 'stove', 'table', 'counter']
    medium_value_locs = ['receptacle', 'drawer', 'cabinet', 'shelf']
    low_value_locs = ['bed', 'sofa', 'chair', 'floor']
    
    if 'you are at' in next_lower or 'arrived at' in next_lower:
        if any(loc in next_lower for loc in high_value_locs):
            location_relevance = 0.15
        elif any(loc in next_lower for loc in medium_value_locs):
            location_relevance = 0.08
        elif any(loc in next_lower for loc in low_value_locs):
            location_relevance = 0.03
        else:
            location_relevance = 0.01
        
        if is_unproductive_search:
            location_relevance -= 0.20
        if is_failure:
            location_relevance -= 0.15
    elif 'go to' in action_lower:
        if any(loc in next_lower for loc in high_value_locs):
            location_relevance = 0.08
        elif any(loc in next_lower for loc in medium_value_locs):
            location_relevance = 0.05
        elif any(loc in next_lower for loc in low_value_locs):
            location_relevance = 0.02
        else:
            location_relevance = 0.01
        
        if is_unproductive_search:
            location_relevance -= 0.15
        if is_failure:
            location_relevance -= 0.12
    
    # Measure goal progress - only reward actions with correct object
    if 'take' in action_lower and ('holding' in next_lower or 'inventory' in next_lower):
        if target_object:
            if target_object in next_lower and target_object in action_lower:
                # Taking the correct target object
                goal_progress += 0.55
                object_relevance += 0.45
            elif target_object in next_lower:
                # Taking target but not explicitly in action
                goal_progress += 0.35
                object_relevance += 0.25
            else:
                # Taking wrong object - penalize
                goal_progress -= 0.10
                object_relevance -= 0.15
        else:
            goal_progress += 0.15
            object_relevance += 0.05
    
    if 'put' in action_lower and ('on' in next_lower or 'in' in next_lower):
        if target_object and target_object in next_lower:
            goal_progress += 0.60
            if target_object in action_lower:
                object_relevance += 0.35
        elif target_object:
            # Putting wrong object
            goal_progress -= 0.15
            object_relevance -= 0.20
        else:
            goal_progress += 0.20
    
    if 'clean' in action_lower and ('clean' in next_lower or 'polished' in next_lower):
        if target_object and target_object in next_lower:
            goal_progress += 0.25
            object_relevance += 0.15
        else:
            goal_progress += 0.05
    
    if 'heat' in action_lower and ('heated' in next_lower or 'warm' in next_lower):
        if target_object and target_object in next_lower:
            goal_progress += 0.25
            object_relevance += 0.15
        else:
            goal_progress += 0.05
    
    if 'open' in action_lower and ('opened' in next_lower or 'open' in next_lower):
        goal_progress += 0.08
        object_relevance += 0.04
    
    if 'close' in action_lower and ('closed' in next_lower or 'close' in next_lower):
        goal_progress += 0.05
        object_relevance += 0.02
    
    # Assess state change quality
    state_change_quality = 0.0
    if next_lower != state_lower:
        if 'holding' in next_lower and 'holding' not in state_lower:
            state_change_quality += 0.20
        if 'on' in next_lower and 'on' not in state_lower:
            state_change_quality += 0.15
        if 'in' in next_lower and 'in' not in state_lower:
            state_change_quality += 0.15
        if 'you are at' in next_lower and 'you are at' not in state_lower:
            state_change_quality += 0.05
        if 'opened' in next_lower and 'opened' not in state_lower:
            state_change_quality += 0.04
        if 'cleaned' in next_lower and 'cleaned' not in state_lower:
            state_change_quality += 0.05
        if 'heated' in next_lower and 'heated' not in state_lower:
            state_change_quality += 0.05
    else:
        state_change_quality -= 0.30
    
    # Calculate task proximity - more conservative, heavily penalize wrong object
    task_proximity = 0.0
    if is_task_complete:
        task_proximity = 1.0
    else:
        if is_holding_target:
            task_proximity += 0.30
        elif is_holding_wrong_object:
            task_proximity -= 0.25  # Holding wrong object is bad
        elif holding_something:
            task_proximity += 0.05
        
        if 'put' in action_lower and target_object and target_object in action_lower:
            task_proximity += 0.15
        elif 'take' in action_lower and target_object and target_object in action_lower:
            task_proximity += 0.12
        
        # Location bonuses - reduced
        if any(loc in next_lower for loc in ['sink', 'stove']):
            task_proximity += 0.05
        if any(loc in next_lower for loc in ['table', 'counter']):
            task_proximity += 0.03
        if any(loc in next_lower for loc in ['receptacle', 'drawer', 'cabinet', 'shelf']):
            task_proximity += 0.02
        
        if 'opened' in next_lower:
            task_proximity += 0.03
        if 'cleaned' in next_lower or 'heated' in next_lower:
            task_proximity += 0.05
        
        if target_object and target_object in next_lower:
            task_proximity += 0.05
    
    # Apply efficiency penalties
    if action_lower.startswith('look') or action_lower.startswith('examine'):
        if 'look' in state_lower or 'examine' in state_lower:
            efficiency -= 0.25
        else:
            efficiency -= 0.05
    
    if 'inventory' in action_lower:
        if 'holding' in state_lower:
            efficiency -= 0.15
        else:
            efficiency -= 0.04
    
    if 'go to' in action_lower:
        if 'you are at' in state_lower and 'you are at' in next_lower:
            state_after = next_lower.split('you are at')[-1][:50] if 'you are at' in next_lower else ''
            state_curr = state_lower.split('you are at')[-1][:50] if 'you are at' in state_lower else ''
            if state_after == state_curr:
                efficiency -= 0.25
    
    if 'put' in action_lower and 'holding' not in state_lower:
        efficiency -= 0.25
    if 'take' in action_lower and 'holding' in state_lower:
        efficiency -= 0.20
    if 'open' in action_lower and 'opened' in state_lower:
        efficiency -= 0.20
    if 'close' in action_lower and 'closed' in state_lower:
        efficiency -= 0.20
    
    if is_failure:
        efficiency -= 0.40
    
    if is_unproductive_search:
        efficiency -= 0.40
    
    if is_loop:
        efficiency -= 0.45
    
    # Strong penalty for holding wrong object
    if is_holding_wrong_object:
        efficiency -= 0.35
    
    # Object mismatch penalty - primary discriminator
    if target_object and not action_matches_target and not is_task_complete:
        if 'take' in action_lower or 'put' in action_lower:
            object_mismatch_penalty = -0.30
        elif 'clean' in action_lower or 'heat' in action_lower:
            object_mismatch_penalty = -0.20
    
    # Compute total score
    total = goal_progress + action_validity + efficiency + location_relevance + state_change_quality + task_proximity + object_relevance + object_mismatch_penalty
    
    # Enforce bounds
    if total > 0.98:
        total = 0.98
    if total < -1.0:
        total = -1.0
    
    return total, {
        "goal_progress": goal_progress,
        "action_validity": action_validity,
        "efficiency": efficiency,
        "location_relevance": location_relevance,
        "state_change_quality": state_change_quality,
        "task_proximity": task_proximity,
        "object_relevance": object_relevance,
        "object_mismatch_penalty": object_mismatch_penalty,
    }