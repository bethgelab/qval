import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for taking action in state resulting in next_state.
    
    Heuristics based on:
    - Action validity and relevance to task
    - Progress indicators in next_state vs state
    - Goal completion signals
    """
    
    # Base value for neutral/uninformative actions
    base_value = 0.3
    
    # Action keywords that typically make progress
    progress_keywords = ['move', 'go to', 'pick up', 'put down', 'take', 'place', 'drop', 
                         'clean', 'open', 'close', 'clean', 'wash', 'dry']
    
    # Success indicators in next state
    success_indicators = ['success', 'done', 'completed', 'goal reached', 'task completed',
                          'finished', 'success', 'reached', 'at', 'in', 'on']
    
    # Negative indicators
    negative_indicators = ['fail', 'error', 'blocked', 'cannot', 'unable', 'stuck', 
                           'already', 'not', 'no', 'empty']
    
    # Check if action appears to be a valid house action
    has_progress_action = any(kw in action.lower() for kw in progress_keywords)
    
    # Check for success indicators
    next_success = any(ind in next_state.lower() for ind in success_indicators)
    state_success = any(ind in state.lower() for ind in success_indicators)
    
    # Check for negative indicators in next state
    next_negative = any(ind in next_state.lower() for ind in negative_indicators)
    
    # Check if state already indicates completion
    state_completed = any(ind in state.lower() for ind in ['goal reached', 'task completed', 'done', 'success'])
    
    # Check if next state indicates completion
    next_completed = any(ind in next_state.lower() for ind in ['goal reached', 'task completed', 'done', 'success'])
    
    # Heuristic scoring
    if next_completed:
        # High reward for reaching goal
        return 0.98
    
    if state_success and not next_success:
        # Already at goal but no progress - action may be redundant
        return 0.5
    
    if next_success and not state_success:
        # Clear progress toward goal
        return 0.85
    
    if next_negative:
        # Action led to negative outcome
        return 0.1
    
    if has_progress_action:
        # Action appears valid
        return 0.6
    
    if next_state == state:
        # No state change - action had no effect
        return 0.2
    
    # Default estimate based on action type
    if has_progress_action:
        return 0.55
    else:
        return 0.4
    
    return base_value