import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value range: 0.0 to 1.0
    q_value = 0.5
    
    # Convert to lowercase for consistent matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Feature 1: Action type analysis
    # Productive actions that typically progress toward goals
    productive_actions = ['put', 'take', 'clean', 'turn on', 'turn off', 'heat', 'cool', 'open', 'close', 'use', 'examine']
    navigation_actions = ['go to', 'walk to', 'move to']
    
    action_is_productive = any(act in action_lower for act in productive_actions)
    action_is_navigation = any(nav in action_lower for nav in navigation_actions)
    
    if action_is_productive:
        q_value += 0.15
    elif action_is_navigation:
        q_value += 0.05
    
    # Feature 2: State contains goal-like indicators
    # Objects in target locations or desired states
    goal_indicators = ['on the', 'in the', 'at the', 'completed', 'success', 'done', 'finished']
    num_goal_indicators = sum(1 for ind in goal_indicators if ind in state_lower)
    
    if num_goal_indicators >= 2:
        q_value += 0.2
    elif num_goal_indicators == 1:
        q_value += 0.1
    
    # Feature 3: Progress detection - compare state to next_state
    # Check if next state shows new object placements or state changes
    state_words = set(re.findall(r'\b\w+\b', state_lower))
    next_state_words = set(re.findall(r'\b\w+\b', next_state_lower))
    
    new_words = next_state_words - state_words
    if len(new_words) > 0:
        # State changed, which could indicate progress
        q_value += 0.05
    
    # Feature 4: Check for error or failure indicators
    error_indicators = ['cannot', 'unable', 'not', 'nothing', 'empty', 'already', 'nothing here', 'no']
    has_error = any(err in next_state_lower for err in error_indicators)
    
    if has_error:
        q_value -= 0.2
    
    # Feature 5: Object manipulation detection
    # Actions involving specific objects are more likely to be goal-directed
    object_patterns = ['the ', 'a ', 'an ']
    has_object_reference = any(pattern in action_lower for pattern in object_patterns)
    
    if has_object_reference:
        q_value += 0.1
    
    # Feature 6: State completeness - longer descriptions may indicate more progress
    state_word_count = len(state_lower.split())
    if state_word_count > 50:
        q_value += 0.05
    elif state_word_count < 20:
        q_value -= 0.05
    
    # Feature 7: Check for specific ALFWorld success patterns
    success_patterns = ['goal achieved', 'task completed', 'successfully', 'you have']
    has_success = any(pat in next_state_lower for pat in success_patterns)
    
    if has_success:
        q_value = 0.95
    
    # Clamp Q-value to reasonable range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value