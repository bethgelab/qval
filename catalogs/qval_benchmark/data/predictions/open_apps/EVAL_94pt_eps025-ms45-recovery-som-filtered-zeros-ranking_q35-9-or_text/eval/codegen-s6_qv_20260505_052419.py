def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for a given state, action, and next_state in OpenApps environment.
    Higher values indicate actions that appear to make progress toward goal completion.
    """
    import re
    
    # Count bid tags to measure state complexity
    def count_bids(text):
        return len(re.findall(r'bid\s*\d+', text))
    
    state_bids = count_bids(state)
    next_bids = count_bids(next_state)
    
    # Check if state meaningfully changed
    state_changed = state != next_state
    
    # Extract action type
    action_type = action.split('(')[0].strip()
    
    # Base estimate based on action type and state change
    if action_type == 'noop':
        base_value = 0.2 if state_changed else 0.4
    elif action_type == 'scroll':
        base_value = 0.4 if next_bids > state_bids else 0.2
    else:
        base_value = 0.5 if state_changed else 0.2
    
    # Adjust based on complexity change
    if next_bids > state_bids:
        base_value += 0.1
    elif next_bids < state_bids:
        base_value -= 0.1
    
    return max(0.0, min(1.0, base_value))