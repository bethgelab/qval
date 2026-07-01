def signal_function(state: str, action: str, next_state: str) -> float:
    import re

    # Score based on whether the action is meaningful (not a noop)
    meaningful_action = action not in ['noop', 'noop(']
    action_score = 1.0 if meaningful_action else 0.3

    # Check for goal achievement indicators in next_state
    goal_indicators = ['completed', 'success', 'done', 'saved', 'created',
                       'added', 'message sent', 'event created', 'task completed',
                       'goal achieved', 'successfully']
    goal_found = any(indicator in next_state.lower() for indicator in goal_indicators)
    goal_score = 1.0 if goal_found else 0.0

    # Check for error indicators in next_state
    error_indicators = ['error', 'failed', 'invalid', 'cannot', 'unable', 'exception',
                        'invalid input', 'not found', 'missing']
    error_found = any(indicator in next_state.lower() for indicator in error_indicators)
    error_score = 0.0 if error_found else 1.0

    # Check if state changed meaningfully (indicates progress)
    state_changed = state != next_state
    progression_score = 1.0 if state_changed else 0.5

    # Count bid tags (interactive elements) as indicators of UI state
    bid_count_state = len(re.findall(r'bid\s*\d+', state))
    bid_count_next = len(re.findall(r'bid\s*\d+', next_state))
    element_score = 0.5 + (bid_count_next - bid_count_state) * 0.1
    element_score = max(0.0, min(1.0, element_score))

    # Check if next_state is shorter (potential completion/cleanup)
    length_diff = len(next_state) - len(state)
    if length_diff < -50:
        length_score = 0.8
    elif length_diff < -20:
        length_score = 0.6
    elif length_diff < 0:
        length_score = 0.5
    elif length_diff > 50:
        length_score = 0.3
    else:
        length_score = 0.5

    # Combine scores with weights reflecting importance
    q_value = (action_score * 0.15 +
               goal_score * 0.40 +
               error_score * 0.20 +
               progression_score * 0.10 +
               element_score * 0.10 +
               length_score * 0.05)

    return round(q_value, 4)