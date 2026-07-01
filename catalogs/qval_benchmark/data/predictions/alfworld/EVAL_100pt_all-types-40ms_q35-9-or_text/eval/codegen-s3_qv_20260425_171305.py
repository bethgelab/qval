import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates Q-value based on text analysis of state, action, and next state.
    Returns a float between 0 and 1 representing expected future return.
    """
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Heuristic scoring factors
    score = 0.0
    
    # Factor 1: Progress indicators in next state vs current state
    # Look for positive progression signals
    progress_keywords = ['done', 'completed', 'success', 'goal reached', 'finished', 
                         'placed', 'moved', 'clean', 'pick up', 'put in', 'grab', 'take']
    
    state_progress = sum(1 for kw in progress_keywords if kw in state_lower)
    next_state_progress = sum(1 for kw in progress_keywords if kw in next_state_lower)
    
    # Bonus if next state shows more progress
    if next_state_progress > state_progress:
        score += 0.2
    elif next_state_progress == state_progress:
        score += 0.1
    
    # Factor 2: Check if action seems appropriate for the task
    # Look for common action patterns that indicate progress
    action_progress_indicators = ['go', 'move', 'pick', 'take', 'put', 'place', 
                                  'clean', 'open', 'close', 'drop', 'hold', 'grab']
    
    if any(indicator in action_lower for indicator in action_progress_indicators):
        score += 0.15
    
    # Factor 3: Check for goal-related state changes
    # If next state mentions goal completion or object in correct location
    goal_completion_signals = ['goal', 'task', 'objective', 'complete', 'finished', 
                               'success', 'done', 'reached', 'in the']
    
    next_goal_signals = sum(1 for kw in goal_completion_signals if kw in next_state_lower)
    state_goal_signals = sum(1 for kw in goal_completion_signals if kw in state_lower)
    
    if next_goal_signals > state_goal_signals:
        score += 0.15
    
    # Factor 4: Check for error or failure signals
    failure_keywords = ['failed', 'error', 'cannot', 'not able', 'blocked', 'unreachable']
    
    if any(fail in next_state_lower for fail in failure_keywords):
        score -= 0.3
    elif any(fail in state_lower for fail in failure_keywords):
        score -= 0.15
    
    # Factor 5: Check for object location consistency
    # If next state shows object in expected location
    location_keywords = ['on the', 'in the', 'at the', 'under the', 'beside the', 
                         'near the', 'behind the', 'inside the', 'outside the',
                         'cabinet', 'table', 'counter', 'shelf', 'floor', 'rug',
                         'kitchen', 'bedroom', 'bathroom', 'living room', 'hallway']
    
    state_location_count = sum(1 for kw in location_keywords if kw in state_lower)
    next_state_location_count = sum(1 for kw in location_keywords if kw in next_state_lower)
    
    if next_state_location_count >= state_location_count:
        score += 0.1
    
    # Factor 6: Check for object manipulation success
    # Look for object status changes that indicate success
    object_status_keywords = ['clean', 'dry', 'full', 'empty', 'open', 'closed',
                              'broken', 'fixed', 'charged', 'unlocked', 'locked']
    
    state_status_count = sum(1 for kw in object_status_keywords if kw in state_lower)
    next_state_status_count = sum(1 for kw in object_status_keywords if kw in next_state_lower)
    
    if next_state_status_count > state_status_count:
        score += 0.1
    
    # Factor 7: Step efficiency consideration
    # If state mentions steps or time, factor that in
    step_keywords = ['steps', 'time', 'turn', 'remaining', 'limit']
    
    state_step_count = sum(1 for kw in step_keywords if kw in state_lower)
    next_state_step_count = sum(1 for kw in step_keywords if kw in next_state_lower)
    
    if next_state_step_count > state_step_count:
        score -= 0.05
    
    # Normalize score to [0, 1] range
    score = max(0.0, min(1.0, score))
    
    return round(score, 4)