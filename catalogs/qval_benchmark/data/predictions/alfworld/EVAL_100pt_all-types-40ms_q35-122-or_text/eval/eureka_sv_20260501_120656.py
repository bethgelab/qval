def signal_function(state: str):
    import re
    
    state_lower = state.lower()
    
    # Check for explicit success indicators - return 1.0 immediately
    success_patterns = [
        r'task completed', r'you have completed the task', r'congratulations',
        r'task succeeded', r'success', r'thank you for completing',
        r'you have completed', r'completed the task', r'the task is complete',
        r'the task has been completed', r'you have successfully',
        r'task is done', r'task finished', r'congrats'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, state_lower):
            return 1.0, {
                "completion_bonus": 1.0,
                "step_penalty": 0.0,
                "object_bonus": 0.0,
                "room_bonus": 0.0,
                "progress_bonus": 0.0,
                "action_failure_penalty": 0.0,
                "exploration_bonus": 0.0,
                "object_discovery_bonus": 0.0,
                "empty_search_penalty": 0.0
            }
    
    # Check for terminal failure (step limit reached)
    terminal_penalty = 0.0
    failure_patterns = [
        r'too many steps', r'out of steps', r'maximum steps',
        r'step limit exceeded', r'no more steps', r'steps exceeded',
        r'no more actions available', r'episode ended', r'time out',
        r'maximum number of steps', r'you have exceeded'
    ]
    
    for pattern in failure_patterns:
        if re.search(pattern, state_lower):
            terminal_penalty = -0.8
            break
    
    # Extract step information with more flexible patterns
    current_step = 0
    total_steps = 40
    
    patterns = [
        r'step\s*(\d+)\s*of\s*(\d+)',
        r'step\s*(\d+)\s*/\s*(\d+)',
        r'step\s*(\d+)\s*[-:]\s*(\d+)',
        r'(\d+)\s*/\s*(\d+)\s*steps?',
        r'\[(\d+)/(\d+)\]',
        r'step\s+(\d+)',
        r'(\d+)\s+step',
        r'(\d+)\s+/\s+(\d+)',
        r'step\s+(\d+)\s*$',
    ]
    
    for pattern in patterns:
        step_match = re.search(pattern, state_lower)
        if step_match:
            try:
                groups = step_match.groups()
                if len(groups) >= 2:
                    current_step = int(groups[0])
                    total_steps = int(groups[1])
                elif len(groups) == 1:
                    current_step = int(groups[0])
                break
            except (ValueError, IndexError):
                pass
    
    # Calculate remaining steps
    remaining_steps = max(0, total_steps - current_step)
    
    # Step penalty - linear decay based on steps used
    step_penalty = -0.015 * current_step
    
    # Room detection with improved patterns
    room_bonus = 0.0
    room_keywords = {
        'kitchen': 0.15,
        'bedroom': 0.12,
        'living room': 0.10,
        'bathroom': 0.10,
        'garage': 0.08,
        'office': 0.08,
        'dining room': 0.10,
        'hallway': 0.05,
        'diningroom': 0.10,
        'livingroom': 0.12,
    }
    
    # Check room keywords in state
    for room, bonus in room_keywords.items():
        if room in state_lower:
            room_bonus = bonus
            break
    
    # Also check for room context patterns
    if room_bonus == 0:
        room_context_patterns = [
            r'you are in the\s+(\w+)',
            r'you are in\s+(\w+(?:\s+\w+)?)',
            r'you see the\s+(\w+)',
            r'you see a\s+(\w+)',
            r'looking at the\s+(\w+)',
            r'you enter the\s+(\w+)',
        ]
        for pat in room_context_patterns:
            match = re.search(pat, state_lower)
            if match:
                potential_room = match.group(1).lower()
                if potential_room in room_keywords:
                    room_bonus = room_keywords[potential_room]
                    break
    
    # Inventory check with improved patterns
    has_object = False
    held_objects = []
    
    inv_patterns = [
        r'holding:\s*(.+?)(?:\n|$)',
        r'you are holding:\s*(.+?)(?:\n|$)',
        r'carrying:\s*(.+?)(?:\n|$)',
        r'in hand:\s*(.+?)(?:\n|$)',
        r'you have:\s*(.+?)(?:\n|$)',
        r'inventory:\s*(.+?)(?:\n|$)',
        r'your inventory:\s*(.+?)(?:\n|$)',
        r'you hold\s+(?:the\s+)?(\w+)',
        r'holding\s+(?:the\s+)?(\w+)',
        r'you are holding\s+(\w+)',
        r'holding\s+(\w+)',
    ]
    
    for inv_pattern in inv_patterns:
        inv_match = re.search(inv_pattern, state_lower)
        if inv_match:
            inv_content = inv_match.group(1).strip() if inv_match.groups() else inv_match.group(0)
            if inv_content and 'nothing' not in inv_content and 'empty' not in inv_content:
                has_object = True
                held_objects = re.findall(r'\b(\w+)\b', inv_content)
                break
    
    # Extract target object from task description
    target_object = None
    task_patterns = [
        r'put\s+the?\s+([\w]+)\s+in\s+the?\s+[\w]+',
        r'put\s+the?\s+([\w]+)\s+on\s+the?\s+[\w]+',
        r'find\s+the?\s+([\w]+)',
        r'take\s+the?\s+([\w]+)',
        r'place\s+the?\s+([\w]+)',
        r'(\w+)\s+in\s+the',
        r'(\w+)\s+on\s+the',
        r'put\s+the?\s+([\w]+)',
        r'the?\s+([\w]+)\s+in',
        r'the?\s+([\w]+)\s+on',
        r'pick\s+up\s+the?\s+([\w]+)',
        r'grab\s+the?\s+([\w]+)',
        r'(\w+)\s+and\s+put',
        r'find\s+([\w]+)',
        r'take\s+([\w]+)',
    ]
    
    for pattern in task_patterns:
        task_match = re.search(pattern, state_lower)
        if task_match:
            target_object = task_match.group(1).strip()
            break
    
    # Object bonus - this should be the main driver
    object_bonus = 0.0
    holding_correct = False
    holding_wrong = False
    
    if target_object and has_object:
        target_words = target_object.lower().split()
        match_found = False
        for held_word in held_objects:
            held_lower = held_word.lower()
            for target_word in target_words:
                if target_word == held_lower or held_lower == target_word:
                    match_found = True
                    break
                if target_word in held_lower and len(target_word) >= 3:
                    match_found = True
                    break
                if held_lower in target_word and len(held_lower) >= 3:
                    match_found = True
                    break
            if match_found:
                break
        
        if match_found:
            holding_correct = True
            object_bonus = 0.45  # Significant reward for correct object
        else:
            holding_wrong = True
            object_bonus = 0.05  # Small reward for wrong object
    elif has_object and not target_object:
        object_bonus = 0.10  # Holding something when we don't know target
    elif target_object and not has_object:
        object_bonus = 0.05  # We know target but don't have it yet
    
    # Action failure check
    action_failed = bool(re.search(
        r'nothing happens|cannot|invalid|error|failed|not possible|no such|does not exist|not here|not found|does not respond|is not available|is empty|nothing to|cannot be found|you cannot|you can not|cannot be|can not be|nothing in|no object|no such object', 
        state_lower
    ))
    
    action_failure_penalty = -0.35 if action_failed else 0.0
    
    # Progress bonus - reduced weights to avoid over-optimism
    progress_bonus = 0.0
    action_keywords = ['put', 'place', 'move', 'take', 'find', 'go to', 'open', 
                       'close', 'clean', 'heat', 'cool', 'put in', 'put on', 
                       'place on', 'place in', 'pick up', 'grab', 'lift', 'look at',
                       'examine', 'check', 'search', 'find the', 'take the',
                       'receptacle', 'container', 'sink', 'stove', 'microwave',
                       'fridge', 'cabinet', 'drawer', 'shelf', 'table', 'desk']
    keyword_count = sum(1 for kw in action_keywords if kw in state_lower)
    progress_bonus += 0.012 * min(keyword_count, 8)
    
    # Interaction patterns - reduced bonus
    interaction_patterns = [
        r'you pick up', r'you take', r'you grab', r'you hold',
        r'you put', r'you place', r'you move',
        r'you open', r'you close', r'you clean',
        r'you heat', r'you cool', r'you look at',
        r'picked up', r'taken', r'grabbed', r'holding',
        r'put', r'placed', r'moved',
        r'opened', r'closed', r'cleaned',
        r'heated', r'cooled', r'you are in', r'you see'
    ]
    interaction_count = sum(1 for pat in interaction_patterns if re.search(pat, state_lower))
    progress_bonus += 0.008 * min(interaction_count, 6)
    
    # Completion bonus - activated when holding correct object in right room
    completion_bonus = 0.0
    if holding_correct and room_bonus > 0:
        # More remaining steps = more potential to complete
        completion_ratio = min(1.0, remaining_steps / 10.0)
        completion_bonus = 0.30 * completion_ratio
    
    # Exploration bonus - reduced
    exploration_bonus = 0.0
    location_patterns = [
        r'you are in', r'you see', r'you can see', r'there is', r'there are',
        r'you look at', r'examining', r'looking at', r'you enter',
        r'walked to', r'arrived at', r'entered the', r'you go to'
    ]
    location_count = sum(1 for pat in location_patterns if re.search(pat, state_lower))
    exploration_bonus = 0.01 * min(location_count, 4)
    
    # Object discovery bonus - reduced weights
    object_discovery_bonus = 0.0
    object_keywords = ['apple', 'banana', 'book', 'bowl', 'box', 'cup', 'candle', 
                       'cloth', 'creditcard', 'desklamp', 'egg', 'keychain', 'ladle',
                       'microwave', 'plate', 'potato', 'receptacle', 'saltshaker',
                       'soapbottle', 'spatula', 'spraybottle', 'statue', 'tomato',
                       'vase', 'watch', 'butterknife', 'dishes', 'pan', 'pan',
                       'kettle', 'coffeemachine', 'toaster', 'sinkbasin', 'fridge',
                       'cabinet', 'drawer', 'shelf', 'table', 'desk', 'sink']
    for obj_kw in object_keywords:
        if obj_kw in state_lower:
            object_discovery_bonus += 0.008
    object_discovery_bonus = min(object_discovery_bonus, 0.10)
    
    # Empty search penalty - activate when searching without progress
    empty_search_penalty = 0.0
    if not has_object and not holding_correct:
        if location_count >= 4 and keyword_count <= 4:
            empty_search_penalty = -0.25
        elif location_count >= 3:
            empty_search_penalty = -0.15
    
    # Calculate total value
    total = (step_penalty + object_bonus + room_bonus + completion_bonus + 
             progress_bonus + action_failure_penalty + terminal_penalty + 
             exploration_bonus + object_discovery_bonus + empty_search_penalty)
    
    # Cap total at 0.75 for non-terminal states to avoid saturation
    if terminal_penalty == 0 and action_failure_penalty == 0:
        total = min(total, 0.75)
    else:
        total = min(total, 1.0)
    
    return total, {
        "step_penalty": step_penalty,
        "object_bonus": object_bonus,
        "room_bonus": room_bonus,
        "completion_bonus": completion_bonus,
        "progress_bonus": progress_bonus,
        "action_failure_penalty": action_failure_penalty,
        "terminal_penalty": terminal_penalty,
        "exploration_bonus": exploration_bonus,
        "object_discovery_bonus": object_discovery_bonus,
        "empty_search_penalty": empty_search_penalty
    }