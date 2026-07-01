def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    score = 0.0
    
    # Action quality score - valid manipulation actions
    valid_actions = ['take', 'put', 'clean', 'cook', 'heat', 'wash', 'slice', 
                     'eat', 'dip', 'add', 'mix', 'open', 'close', 'turn on', 
                     'turn off', 'go to', 'move', 'examine', 'look']
    action_lower = action.lower()
    if any(kw in action_lower for kw in valid_actions):
        score += 0.2
    
    # Goal-related keywords in state indicate task context
    goal_keywords = ['goal', 'target', 'place', 'move', 'clean', 'cook', 
                     'heat', 'eat', 'wash', 'slice', 'dip', 'add', 'mix',
                     'open', 'close', 'turn', 'move', 'go to', 'examine', 'look']
    state_lower = state.lower()
    if any(kw in state_lower for kw in goal_keywords):
        score += 0.2
    
    # Progress indicator - check if next state shows completion markers
    completion_markers = ['complete', 'done', 'success', 'placed', 'found', 
                          'completed', 'finished', 'success']
    if any(marker in next_state_lower for marker in completion_markers):
        score += 0.4
    
    # State change detection - compare tokens between states
    state_tokens = set(state_lower.split())
    next_state_tokens = set(next_state_lower.split())
    
    # Significant token changes indicate progress
    token_diff = len(state_tokens ^ next_state_tokens)
    if token_diff >= 5:
        score += 0.3
    elif token_diff >= 3:
        score += 0.2
    elif token_diff >= 1:
        score += 0.1
    
    # Location-based progress - check for room transitions
    rooms = ['kitchen', 'bedroom', 'living room', 'bathroom', 'garage', 
             'dining room', 'study', 'hallway', 'office']
    if any(room in state_lower for room in rooms) and any(room in next_state_lower for room in rooms):
        score += 0.1
    
    # Object state changes (clean, cooked, heated, etc.)
    object_states = ['clean', 'dirty', 'cooked', 'raw', 'heated', 'cold', 
                     'wet', 'dry', 'sliced', 'whole', 'open', 'closed']
    if any(state in state_lower for state in object_states):
        score += 0.1
    
    # Penalty for no progress (same state)
    if state_lower == next_state_lower:
        score -= 0.1
    
    # Ensure score is within valid range
    score = max(0.0, min(1.0, score))
    
    return score