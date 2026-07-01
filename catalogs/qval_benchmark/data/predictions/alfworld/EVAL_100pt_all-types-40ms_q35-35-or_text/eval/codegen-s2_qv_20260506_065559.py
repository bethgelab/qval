import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for ALFWorld based on state features and action progress.
    Returns a float in [0, 1] representing expected discounted cumulative reward.
    """
    score = 0.0
    
    # Check for task completion in next_state
    completion_patterns = ['success', 'completed', 'done', 'finished', 'task complete']
    for pattern in completion_patterns:
        if pattern in next_state.lower():
            return 1.0
    
    # Check for failure/error indicators
    failure_patterns = ['error', 'fail', 'invalid', 'cannot', 'impossible', 'no', 'not']
    for pattern in failure_patterns:
        if pattern in next_state.lower():
            return 0.0
    
    # Action type scoring
    action_lower = action.lower()
    
    # High-value actions (directly manipulate objects toward goal)
    high_value_actions = ['put', 'take', 'move', 'carry', 'pick', 'grab']
    for act in high_value_actions:
        if act in action_lower:
            score += 0.15
    
    # Medium-value actions (navigation, container manipulation)
    medium_value_actions = ['go', 'walk', 'open', 'close', 'clean', 'wash', 'cook', 'heat', 'cool']
    for act in medium_value_actions:
        if act in action_lower:
            score += 0.1
    
    # Low-value actions (status checks, confirmations)
    low_value_actions = ['look', 'check', 'see', 'read', 'say', 'hello']
    for act in low_value_actions:
        if act in action_lower:
            score += 0.05
    
    # Progress indicators in next_state
    location_patterns = ['on the', 'in the', 'at the', 'by the', 'near the']
    for pattern in location_patterns:
        if pattern in next_state.lower():
            score += 0.08
    
    # Container state changes
    container_patterns = ['opened', 'closed', 'inside', 'inside of', 'in', 'out']
    for pattern in container_patterns:
        if pattern in next_state.lower():
            score += 0.06
    
    # Object state changes (cleaned, cooked, heated, etc.)
    state_change_patterns = ['cleaned', 'washed', 'cooked', 'heated', 'cooled', 'chopped', 'sliced']
    for pattern in state_change_patterns:
        if pattern in next_state.lower():
            score += 0.12
    
    # Inventory changes (holding, picked, grabbed)
    inventory_patterns = ['holding', 'picked', 'grabbed', 'taken', 'carrying']
    for pattern in inventory_patterns:
        if pattern in next_state.lower():
            score += 0.1
    
    # Compare state complexity as progress indicator
    state_words = len(state.split())
    next_state_words = len(next_state.split())
    
    if next_state_words > state_words:
        score += 0.05
    elif next_state_words < state_words:
        score -= 0.05
    
    # Check for room/location mentions in navigation actions
    if 'go' in action_lower or 'walk' in action_lower:
        if 'room' in next_state.lower() or 'to' in next_state.lower():
            score += 0.08
    
    # Normalize to [0, 1] range
    score = max(0.0, min(1.0, score))
    
    # Slight penalty for very low scores to encourage exploration
    if score < 0.1:
        score = 0.1
    
    return score