import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for error indicators in next_state
    error_patterns = ['nothing happens', 'cannot', 'already', 'not here', 'failed', 'error', 'cannot find']
    has_error = any(pattern in next_state.lower() for pattern in error_patterns)
    
    # Check for success/terminal indicators
    success_patterns = ['done', 'success', 'completed', 'goal achieved', 'task complete']
    is_success = any(pattern in next_state.lower() for pattern in success_patterns)
    
    # Check for meaningful action progress
    progress_patterns = ['took', 'put', 'cleaned', 'heated', 'opened', 'closed', 'picked up', 'placed']
    has_progress = any(pattern in next_state.lower() for pattern in progress_patterns)
    
    # Check for navigation progress
    nav_patterns = ['walked to', 'went to', 'arrived at', 'now at', 'you are now at']
    has_nav_progress = any(pattern in next_state.lower() for pattern in nav_patterns)
    
    # Check if agent is holding an object (intermediate progress)
    holding_patterns = ['holding', 'you are holding', 'in your inventory']
    is_holding = any(pattern in next_state.lower() for pattern in holding_patterns)
    
    # Check if state changed meaningfully
    state_changed = state.lower() != next_state.lower()
    
    # Base Q-value estimate
    if is_success:
        return 1.0
    
    if has_error:
        return -0.4
    
    q_value = 0.0
    
    # Reward for meaningful progress
    if has_progress:
        q_value += 0.35
    if has_nav_progress:
        q_value += 0.25
    
    # Small bonus for holding objects (may be intermediate step)
    if is_holding:
        q_value += 0.15
    
    # Bonus for state change (action had effect)
    if state_changed:
        q_value += 0.1
    
    # Penalty for no state change (action ineffective)
    if not state_changed and not has_error:
        q_value -= 0.2
    
    # Check for task-relevant keywords in action
    action_keywords = ['take', 'put', 'clean', 'heat', 'open', 'close', 'walk', 'go']
    action_has_keyword = any(kw in action.lower() for kw in action_keywords)
    if action_has_keyword:
        q_value += 0.05
    
    # Check if next_state mentions goal-related objects
    goal_keywords = ['cup', 'bottle', 'mug', 'spoon', 'fork', 'knife', 'apple', 'banana']
    if any(kw in next_state.lower() for kw in goal_keywords):
        q_value += 0.05
    
    # Clamp to reasonable range
    return max(-1.0, min(1.0, q_value))