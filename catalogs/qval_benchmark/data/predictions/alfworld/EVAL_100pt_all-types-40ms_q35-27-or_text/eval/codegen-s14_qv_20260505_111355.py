import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for success indicators in next_state
    success_keywords = ['done', 'completed', 'success', 'goal', 'achieved', 'finished']
    is_success = any(kw in next_state.lower() for kw in success_keywords)
    
    if is_success:
        # High value for successful completion
        return 1.0
    
    # Check for error/invalid action indicators
    error_keywords = ['cannot', 'can\'t', 'not', 'error', 'invalid', 'nothing', 'already']
    has_error = any(kw in next_state.lower() for kw in error_keywords)
    
    # Analyze action type
    action_lower = action.lower()
    action_type = 'other'
    
    if 'go' in action_lower or 'walk' in action_lower or 'move' in action_lower:
        action_type = 'navigate'
    elif 'take' in action_lower or 'get' in action_lower:
        action_type = 'take'
    elif 'put' in action_lower or 'place' in action_lower or 'drop' in action_lower:
        action_type = 'put'
    elif 'clean' in action_lower or 'wash' in action_lower:
        action_type = 'clean'
    elif 'open' in action_lower or 'close' in action_lower:
        action_type = 'toggle'
    elif 'heat' in action_lower or 'cook' in action_lower:
        action_type = 'heat'
    elif 'examine' in action_lower or 'look' in action_lower:
        action_type = 'observe'
    
    # Check if state changed meaningfully
    state_normalized = ' '.join(state.lower().split())
    next_normalized = ' '.join(next_state.lower().split())
    
    # Count common words that changed
    state_words = set(state_normalized.split())
    next_words = set(next_normalized.split())
    
    # Words that appeared or disappeared
    new_words = next_words - state_words
    removed_words = state_words - next_words
    
    # Check for progress indicators
    progress_keywords = ['now', 'is', 'became', 'changed', 'success', 'found', 'reached']
    has_progress = any(kw in next_normalized for kw in progress_keywords)
    
    # Calculate state difference score
    state_diff = len(new_words) + len(removed_words)
    
    # Base Q-value estimate
    base_value = 0.0
    
    # Penalize invalid actions
    if has_error:
        base_value = -0.5
    else:
        # Reward valid actions that cause state changes
        if state_diff > 0:
            base_value += 0.3
        
        # Bonus for progress indicators
        if has_progress:
            base_value += 0.2
        
        # Action type bonuses (some actions are more likely to lead to progress)
        if action_type in ['take', 'put']:
            base_value += 0.15
        elif action_type == 'clean':
            base_value += 0.1
        elif action_type == 'navigate':
            base_value += 0.05
        elif action_type == 'observe':
            base_value += 0.02
    
    # Check if action seems to be moving toward goal (heuristic)
    # Look for goal-related words in next_state
    goal_words = ['target', 'destination', 'goal', 'location', 'place']
    goal_proximity = sum(1 for kw in goal_words if kw in next_normalized)
    if goal_proximity > 0:
        base_value += 0.1 * goal_proximity
    
    # Cap the value between -1 and 1
    return max(-1.0, min(1.0, base_value))