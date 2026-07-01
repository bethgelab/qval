import re

def signal_function(state: str) -> float:
    """
    Estimate the state-value V(s) for ALFWorld based on task progress indicators.
    Uses heuristic analysis of state text without simulation or lookahead.
    """
    state_lower = state.lower()
    
    # Check for task completion
    success_indicators = [
        'task completed', 'task is complete', 'success', 'goal achieved',
        'you have successfully', 'task is done', 'completed the task',
        'successfully cleaned', 'successfully moved'
    ]
    if any(ind in state_lower for ind in success_indicators):
        return 1.0
    
    # Check for failure
    failure_indicators = [
        'task failed', 'failed', 'you failed', 'could not', 'unable to',
        'too many attempts', 'max attempts reached'
    ]
    if any(ind in state_lower for ind in failure_indicators):
        return 0.0
    
    # Base value for non-terminal states
    base_value = 0.3
    
    # Check for positive progress indicators
    progress_indicators = [
        'i have', 'i am in', 'the [object]', 'on the', 'in the',
        'near', 'next to', 'beside', 'under', 'above', 'clean', 'ready'
    ]
    progress_count = sum(1 for ind in progress_indicators if ind.replace('[object]', '') in state_lower)
    progress_score = min(0.4, progress_count * 0.05)
    
    # Check for task-relevant objects (common ALFWorld objects)
    object_patterns = [
        'dish', 'broom', 'vacuum', 'lamp', 'trash', 'mop',
        'cup', 'plate', 'bowl', 'towel', 'pillow', 'blanket',
        'remote', 'phone', 'book', 'magazine', 'glasses', 'key',
        'soap', 'shampoo', 'toothbrush', 'toothpaste', 'bottle'
    ]
    object_mentions = sum(1 for obj in object_patterns if obj in state_lower)
    object_score = min(0.2, object_mentions * 0.03)
    
    # Check for location context
    location_patterns = [
        'kitchen', 'bedroom', 'bathroom', 'living room', 'dining room',
        'hallway', 'office', 'garage', 'balcony', 'patio', 'yard'
    ]
    location_mentions = sum(1 for loc in location_patterns if loc in state_lower)
    location_score = min(0.1, location_mentions * 0.02)
    
    # Check for action-related words (indicates active engagement)
    action_words = [
        'pick', 'drop', 'clean', 'move', 'take', 'bring', 'put',
        'wash', 'dry', 'fold', 'iron', 'sweep', 'vacuum', 'clean'
    ]
    action_count = sum(1 for action in action_words if action in state_lower)
    action_score = min(0.15, action_count * 0.02)
    
    # Combine scores
    total_score = base_value + progress_score + object_score + location_score + action_score
    
    # Ensure value is in [0, 1]
    return max(0.0, min(1.0, total_score))