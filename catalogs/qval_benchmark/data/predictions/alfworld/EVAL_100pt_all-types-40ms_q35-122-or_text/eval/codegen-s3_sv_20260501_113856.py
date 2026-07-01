def signal_function(state: str) -> float:
    import re
    
    # Check for explicit success indicators
    success_phrases = ['success', 'completed', 'congratulations', 'task complete', 
                       'you have placed', 'task completed', 'well done']
    for phrase in success_phrases:
        if phrase in state.lower():
            return 1.0
    
    # Check for explicit failure indicators
    failure_phrases = ['failed', 'error', 'cannot', 'impossible', 'not found', 
                       'not available', 'out of range', 'step limit reached']
    for phrase in failure_phrases:
        if phrase in state.lower():
            return 0.0
    
    # Extract step count if available
    step_match = re.search(r'step\s*(\d+)', state.lower())
    current_step = int(step_match.group(1)) if step_match else 0
    max_steps = 40
    remaining_steps = max(0, max_steps - current_step)
    
    # Calculate progress based on task-relevant keywords
    # Look for objects being manipulated (take, put, go, open, close, etc.)
    action_verbs = ['take', 'put', 'go', 'open', 'close', 'clean', 'heat', 'cool', 
                    'slice', 'cook', 'microwave', 'wash', 'fill', 'empty']
    action_count = sum(1 for verb in action_verbs if verb in state.lower())
    
    # Look for location keywords (room names)
    locations = ['kitchen', 'living room', 'bedroom', 'bathroom', 'dining room', 
                 'office', 'garage', 'basement', 'hallway', 'pantry']
    location_count = sum(1 for loc in locations if loc in state.lower())
    
    # Look for container keywords (indicating interaction with receptacles)
    containers = ['drawer', 'cupboard', 'shelf', 'counter', 'table', 'sink', 
                  'fridge', 'microwave', 'oven', 'cabinet', 'basket', 'box']
    container_count = sum(1 for cont in containers if cont in state.lower())
    
    # Look for object keywords (indicating items are being handled)
    objects = ['apple', 'banana', 'tomato', 'egg', 'potato', 'onion', 'lettuce',
               'plate', 'bowl', 'cup', 'mug', 'knife', 'fork', 'spoon', 'cloth',
               'towel', 'book', 'pen', 'paper', 'laptop', 'phone', 'remote',
               'key', 'wallet', 'watch', 'clock', 'lamp', 'tv', 'monitor']
    object_count = sum(1 for obj in objects if obj in state.lower())
    
    # Calculate progress score (0 to 1)
    # More actions, locations, containers, and objects mentioned = more progress
    total_keywords = action_count + location_count + container_count + object_count
    progress_score = min(1.0, total_keywords / 20.0)
    
    # Calculate efficiency score based on remaining steps
    # More remaining steps = better position to complete task
    efficiency_score = remaining_steps / max_steps if max_steps > 0 else 0.0
    
    # Combine progress and efficiency
    # Weight progress more heavily as it's more predictive of eventual success
    value = 0.7 * progress_score + 0.3 * efficiency_score
    
    # Ensure value is in valid range
    value = max(0.0, min(1.0, value))
    
    return value