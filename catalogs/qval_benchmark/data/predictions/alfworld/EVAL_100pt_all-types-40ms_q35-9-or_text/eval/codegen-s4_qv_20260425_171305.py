import re

def signal_function(state: str, action: str, next_state: str) -> float:
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    progress_keywords = ['goal', 'success', 'complete', 'done', 'finished', 'placed', 'clean', 'poured', 'washed', 'opened', 'closed', 'brought', 'delivered']
    regression_keywords = ['error', 'failed', 'wrong', 'unable', 'blocked', 'cannot', 'not', 'missing', 'broken', 'dirty', 'full']
    location_keywords = ['kitchen', 'living room', 'bedroom', 'bathroom', 'hallway', 'dining room', 'study', 'office']
    object_keywords = ['bottle', 'cup', 'glass', 'plate', 'bowl', 'food', 'water', 'milk', 'juice', 'dish', 'soap', 'cleaner', 'towel', 'paper', 'book', 'toy']
    
    state_pos = sum(1 for kw in progress_keywords if kw in state_lower)
    state_neg = sum(1 for kw in regression_keywords if kw in state_lower)
    next_pos = sum(1 for kw in progress_keywords if kw in next_state_lower)
    next_neg = sum(1 for kw in regression_keywords if next_state_lower)
    
    action_relevant = 0
    if any(kw in action_lower for kw in ['take', 'put', 'clean', 'wash', 'move', 'go', 'pick', 'drop', 'bring', 'deliver', 'open', 'close', 'pour', 'fill']):
        action_relevant = 1
    
    location_change = 0
    if any(kw in state_lower for kw in location_keywords) and any(kw in next_state_lower for kw in location_keywords):
        if state_lower != next_state_lower:
            location_change = 1
    
    has_object = any(kw in state_lower for kw in object_keywords)
    has_object_next = any(kw in next_state_lower for kw in object_keywords)
    
    if next_pos > state_pos and next_neg < state_neg:
        q_value = 0.6 + (next_pos * 0.1)
    elif action_relevant and (next_pos >= state_pos or next_neg <= state_neg):
        q_value = 0.4 + (state_pos * 0.1)
    elif location_change and has_object:
        q_value = 0.3 + (state_pos * 0.1)
    elif state_pos > 0:
        q_value = 0.2 + (state_pos * 0.05)
    elif action_relevant:
        q_value = 0.15
    else:
        q_value = 0.05 - (state_neg * 0.02)
    
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value