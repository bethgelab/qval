import re
import math

def signal_function(state: str) -> float:
    """
    Estimate the state-value for ALFWorld environment.
    
    Higher value = more likely to succeed with good efficiency.
    Based on parsing state text for task completion, progress, and difficulty signals.
    """
    state_lower = state.lower()
    
    # Task completion indicators
    completion_patterns = [
        r'task completed', r'task finished', r'task done',
        r'goal achieved', r'successfully', r'completed',
        r'goal reached', r'object placed', r'object moved'
    ]
    completion_count = sum(1 for pattern in completion_patterns if pattern in state_lower)
    
    # Task failure/blocked indicators
    failure_patterns = [
        r'task failed', r'task blocked', r'cannot', r'cannot',
        r'error', r'failed', r'blocked', r'unable', r'not able'
    ]
    failure_count = sum(1 for pattern in failure_patterns if pattern in state_lower)
    
    # Progress action indicators
    progress_patterns = [
        r'move', r'clean', r'put', r'place', r'take', r'pick up',
        r'go to', r'go into', r'go out', r'enter', r'leave',
        r'open', r'close', r'clean', r'wash', r'dry', r'fold'
    ]
    progress_count = sum(1 for pattern in progress_patterns if pattern in state_lower)
    
    # Remaining work indicators
    work_patterns = [
        r'need to', r'must', r'should', r'has to', r'needs',
        r'needs to', r'waiting for', r'needs cleaning', r'needs moving'
    ]
    work_count = sum(1 for pattern in work_patterns if pattern in state_lower)
    
    # Location context (being in relevant location helps)
    location_patterns = [
        r'in the', r'at the', r'on the', r'near', r'close to',
        r'in kitchen', r'in living room', r'in bedroom', r'in bathroom',
        r'in hallway', r'in balcony', r'in garage', r'in study'
    ]
    location_count = sum(1 for pattern in location_patterns if pattern in state_lower)
    
    # Object state indicators (clean/positioned objects are good)
    good_state_patterns = [
        r'clean', r'washed', r'dried', r'folded', r'placed',
        r'moved', r'positioned', r'correct', r'right place'
    ]
    good_state_count = sum(1 for pattern in good_state_patterns if pattern in state_lower)
    
    # Start with neutral value
    value = 0.5
    
    # Adjust for completion (strong positive signal)
    if completion_count > 0:
        value = min(0.95, value + 0.15 * completion_count)
    
    # Adjust for failure (strong negative signal)
    if failure_count > 0:
        value = max(0.05, value - 0.15 * failure_count)
    
    # Adjust for progress made (moderate positive signal)
    if progress_count > 0:
        value = min(0.95, value + 0.05 * progress_count)
    
    # Adjust for remaining work (moderate negative signal)
    if work_count > 0:
        value = max(0.05, value - 0.05 * work_count)
    
    # Adjust for location context (slight positive signal)
    if location_count > 0:
        value = min(0.95, value + 0.02 * location_count)
    
    # Adjust for good object states (slight positive signal)
    if good_state_count > 0:
        value = min(0.95, value + 0.03 * good_state_count)
    
    # Ensure value is bounded
    value = max(0.0, min(1.0, value))
    
    return value