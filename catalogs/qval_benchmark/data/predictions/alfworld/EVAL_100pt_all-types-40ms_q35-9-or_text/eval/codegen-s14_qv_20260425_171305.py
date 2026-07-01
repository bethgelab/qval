import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    
    Returns a float representing the estimated expected return.
    Higher values indicate actions that are more likely to lead to successful task completion.
    """
    
    # Check if task is already complete in current state
    if 'completed' in state.lower() or 'success' in state.lower() or 'done' in state.lower():
        return 0.95
    
    # Check if next state indicates task completion
    if 'completed' in next_state.lower() or 'success' in next_state.lower() or 'done' in next_state.lower():
        return 0.9
    
    # Extract key information from state text
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Identify goal-related keywords
    goal_keywords = ['goal', 'target', 'put', 'move', 'clean', 'fetch', 'pick', 'place', 'find', 'bring']
    progress_keywords = ['closer', 'near', 'reached', 'found', 'picked', 'placed', 'cleaned']
    negative_keywords = ['failed', 'error', 'cannot', 'unable', 'locked', 'blocked', 'broken']
    
    # Count occurrences
    goal_count = sum(state_lower.count(k) for k in goal_keywords) + sum(next_state_lower.count(k) for k in goal_keywords)
    progress_count = sum(state_lower.count(k) for k in progress_keywords) + sum(next_state_lower.count(k) for k in progress_keywords)
    negative_count = sum(state_lower.count(k) for k in negative_keywords) + sum(next_state_lower.count(k) for k in negative_keywords)
    
    # Check if action seems meaningful
    action_lower = action.lower()
    action_keywords = ['go', 'move', 'pick', 'place', 'put', 'clean', 'fetch', 'take', 'drop', 'open', 'close']
    action_meaningful = any(kw in action_lower for kw in action_keywords)
    
    # Check if next state shows improvement (new object found, location changed, etc.)
    state_length = len(state)
    next_state_length = len(next_state)
    
    # Check for location changes
    location_changes = bool(re.search(r'now|current|at|in|on|to|from', next_state_lower))
    
    # Check for step count if available
    step_match = re.search(r'step\s*(\d+)', state_lower)
    steps_remaining = None
    if step_match:
        steps_remaining = 40 - int(step_match.group(1))
    
    # Base Q-value calculation
    base_value = 0.1
    
    # Increase value if action is meaningful
    if action_meaningful:
        base_value += 0.2
    
    # Increase value if next state shows progress indicators
    if progress_count > 0:
        base_value += 0.15
    
    # Increase value if goal keywords appear
    if goal_count > 0:
        base_value += 0.1
    
    # Decrease value if negative indicators present
    if negative_count > 0:
        base_value -= 0.3
    
    # Decrease value if no progress in next state
    if not location_changes and not progress_count > 0:
        base_value -= 0.1
    
    # Cap between -0.5 and 1.0
    q_value = max(-0.5, min(1.0, base_value))
    
    # If we have step information, adjust based on steps remaining
    if steps_remaining is not None:
        if steps_remaining <= 10:
            q_value += 0.15
        elif steps_remaining <= 20:
            q_value += 0.05
    
    return q_value