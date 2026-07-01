import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on state features and action effectiveness.
    Returns a float between 0 and 1 representing expected return.
    """
    
    # Base score for valid action execution
    base_score = 0.3
    
    # Check for task completion indicators in next_state
    completion_patterns = [
        r'task completed', r'completed', r'finished', r'success',
        r'you have', r'congratulations', r'goal', r'done'
    ]
    completion_score = 0.0
    for pattern in completion_patterns:
        if re.search(pattern, next_state.lower()):
            completion_score = 1.0
            break
    
    # Check if action is productive based on action keywords
    productive_actions = [
        r'go to', r'go', r'walk to', r'walk', r'take', r'pick',
        r'put', r'move', r'clean', r'wash', r'cook', r'eat',
        r'refrigerate', r'heat', r'cool', r'store', r'open',
        r'close', r'clean', r'wipe', r'fill', r'empty'
    ]
    action_productive = 0.0
    for pattern in productive_actions:
        if re.search(pattern, action.lower()):
            action_productive = 0.4
            break
    
    # Check if action is counterproductive
    counterproductive_patterns = [
        r'look', r'wait', r'help', r'what', r'where',
        r'why', r'how', r'who', r'when'
    ]
    action_counterproductive = 0.0
    for pattern in counterproductive_patterns:
        if re.search(pattern, action.lower()):
            action_counterproductive = 0.2
            break
    
    # Check progress by comparing state changes
    state_progress = 0.0
    
    # Extract object mentions from states
    object_pattern = r'(?:the\s+)?(?:[a-z]+)\s+(?:that\s+)?(?:is\s+)?(?:in|on|at|under|near|next to|inside|outside|in front of|behind|above|below|between|beside|against|into|onto|from|to|with|by)\s+(?:the\s+)?(?:[a-z]+)'
    state_objects = re.findall(object_pattern, state, re.IGNORECASE)
    next_objects = re.findall(object_pattern, next_state, re.IGNORECASE)
    
    # Check if new objects are mentioned in next_state (progress)
    if next_objects and not state_objects:
        state_progress = 0.2
    elif next_objects:
        # Check for new object-location pairs
        new_pairs = set(next_objects) - set(state_objects)
        if len(new_pairs) > 0:
            state_progress = 0.15 * min(len(new_pairs), 3)
    
    # Check for location-based progress
    location_keywords = ['room', 'kitchen', 'bathroom', 'bedroom', 'living room', 
                        'hallway', 'countertop', 'table', 'desk', 'shelf', 
                        'cabinet', 'drawer', 'refrigerator', 'microwave', 
                        'sink', 'toilet', 'bathtub', 'stove', 'oven']
    state_locations = [loc for loc in location_keywords if loc in state.lower()]
    next_locations = [loc for loc in location_keywords if loc in next_state.lower()]
    
    location_progress = 0.0
    if len(next_locations) > len(state_locations):
        location_progress = 0.1 * (len(next_locations) - len(state_locations))
    
    # Check if action matches state context (validity)
    action_validity = 0.0
    valid_action_contexts = [
        (r'take', r'hold|on the|on a|on the|in the|on the|at the'),
        (r'put', r'in the|on the|onto the|into the|in a|on a'),
        (r'go to', r'room|kitchen|bathroom|bedroom|living room'),
        (r'open', r'the|a'),
        (r'close', r'the|a'),
        (r'clean', r'the|a'),
        (r'wash', r'the|a'),
        (r'fill', r'the|a'),
        (r'empty', r'the|a')
    ]
    
    for action_kw, context_kw in valid_action_contexts:
        if re.search(action_kw, action.lower()) and re.search(context_kw, state.lower()):
            action_validity = 0.15
            break
    
    # Calculate final Q-value estimate
    # Combine all factors with appropriate weights
    q_value = base_score + completion_score + action_productive + state_progress + location_progress + action_validity - action_counterproductive
    
    # Ensure Q-value is in reasonable bounds [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    # Apply slight discount for steps taken (efficiency)
    # Estimate remaining steps based on state complexity
    word_count = len(next_state.split())
    step_penalty = min(0.2, word_count * 0.005)  # More complex state = potentially more steps
    q_value = q_value * (1.0 - step_penalty)
    
    return round(q_value, 4)