import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    The Q-value is based on heuristics identifying progress towards typical goals (creating,
    sending, saving, updating) and observing state transitions.
    """
    
    # Baseline Q-value
    q_value = 0.2
    
    # 1. Analyze the Action
    # Extract bid and potentially the text value from the action string
    click_match = re.search(r"click\('(\d+)'\)", action)
    fill_match = re.search(r"fill\('(\d+)',\s*'(.*?)'\)", action)
    
    if action.startswith("noop"):
        q_value = 0.1
    elif fill_match:
        # Filling a form is generally progress
        q_value = 0.4
    elif click_match:
        bid = click_match.group(1)
        # Try to find the element's label in the state to determine the nature of the click
        # Looking for patterns like [bid=5] Label or bid=5 ... >Label<
        element_pattern = rf"bid={bid}[^>]*>(.*?)<"
        element_match = re.search(element_pattern, state)
        
        if element_match:
            label = element_match.group(1).lower()
            # High-value targets: buttons that commit changes
            if any(word in label for word in ["save", "submit", "send", "create", "add", "confirm", "done"]):
                q_value = 0.7
            # Medium-value targets: buttons that start a process
            elif any(word in label for word in ["new", "compose", "plus", "edit"]):
                q_value = 0.5
        else:
            # Generic click
            q_value = 0.3
    elif action.startswith("scroll"):
        q_value = 0.2
    
    # 2. Analyze the Next State for Success Indicators
    # These keywords strongly suggest the goal has been achieved
    success_keywords = [
        "success", "successfully", "created", "sent", "saved", 
        "updated", "added", "confirmed", "completed"
    ]
    
    next_state_lower = next_state.lower()
    for word in success_keywords:
        if word in next_state_lower:
            # If the state contains success text, the action was likely the final step
            q_value = max(q_value, 0.9)
            break

    # 3. Analyze State Transition
    # If the state didn't change, the action was likely ineffective (stuck)
    if next_state == state:
        q_value *= 0.5
    else:
        # A change in state usually indicates progress
        q_value += 0.1

    # Final clamping to ensure Q-value is within [0, 1]
    return float(max(0.0, min(1.0, q_value)))