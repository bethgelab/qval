def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for state, action, and next_state in ALFWorld.
    """
    import re

    q_value = 0.5

    success_patterns = ['task complete', 'done', 'success', 'completed', 'goal achieved', 'finished']
    if any(re.search(pattern, state.lower()) for pattern in success_patterns):
        q_value = 1.0
        return q_value

    action_keywords = ['move', 'pick', 'place', 'open', 'close', 'clean', 'put', 'take', 'go', 'navigate']
    has_action = any(kw in action.lower() for kw in action_keywords)

    if not has_action:
        q_value = 0.1
        return q_value

    progress_keywords = ['moved', 'placed', 'clean', 'opened', 'closed', 'picked', 'put down', 'took', 'carrying', 'on table', 'in', 'to']
    state_progress = any(kw in state.lower() for kw in progress_keywords)
    next_state_progress = any(kw in next_state.lower() for kw in progress_keywords)

    if next_state_progress and not state_progress:
        q_value = 0.8
    elif next_state_progress and state_progress:
        q_value = 0.7
    elif not next_state_progress and state_progress:
        q_value = 0.3
    else:
        q_value = 0.5

    negative_patterns = ['error', 'fail', 'invalid', 'cannot', 'blocked', 'not able', 'unable']
    if any(re.search(pattern, next_state.lower()) for pattern in negative_patterns):
        q_value = 0.0

    location_keywords = ['kitchen', 'living room', 'bedroom', 'bathroom', 'hallway', 'room', 'location']
    if not any(kw in state.lower() for kw in location_keywords):
        q_value *= 0.8

    return max(0.0, min(1.0, q_value))