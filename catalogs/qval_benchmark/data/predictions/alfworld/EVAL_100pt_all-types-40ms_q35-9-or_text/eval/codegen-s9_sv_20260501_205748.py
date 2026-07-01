import re
import math

def signal_function(state: str) -> float:
    """
    Estimate state-value for ALFWorld based on state text representation.
    Higher values indicate states closer to successful task completion.
    """
    state_lower = state.lower()
    
    # Task completion indicators (highest value boost)
    completion_keywords = [
        'done', 'complete', 'success', 'task complete', 'task completed',
        'finished', 'completed', 'goal reached', 'goal achieved',
        'succeed', 'success'
    ]
    
    # Progress indicators (moderate value boost)
    progress_keywords = [
        'placed', 'cleaned', 'picked', 'holding', 'holding ',
        'in hand', 'on table', 'on floor', 'on bed', 'in box',
        'near', 'at', 'target', 'destination', 'reached',
        'move to', 'put', 'drop', 'throw'
    ]
    
    # Location context (helps assess proximity to goal)
    location_keywords = [
        'kitchen', 'bedroom', 'living room', 'bathroom',
        'hallway', 'door', 'window', 'table', 'floor', 'bed',
        'chair', 'shelf', 'counter', 'cabinet', 'drawer'
    ]
    
    # Negative/error indicators (value penalty)
    error_keywords = [
        'error', 'fail', 'blocked', 'cannot', 'unable',
        'wrong', 'not', 'missing', 'lost', 'cannot find',
        'locked', 'closed', 'full', 'empty', 'broken',
        'already', 'already ', 'done already'
    ]
    
    # Check for task completion
    if any(kw in state_lower for kw in completion_keywords):
        return 0.95
    
    # Check for progress indicators
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    
    # Check for error indicators
    error_count = sum(1 for kw in error_keywords if kw in state_lower)
    
    # Base value starts at 0.5 (neutral)
    value = 0.5
    
    # Adjust based on progress (up to +0.3)
    value += min(progress_count * 0.025, 0.3)
    
    # Adjust based on errors (down to -0.3)
    value -= min(error_count * 0.025, 0.3)
    
    # Clamp to [0, 1]
    value = max(0.0, min(1.0, value))
    
    return value