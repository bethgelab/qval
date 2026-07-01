import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value (expected cumulative reward) for a state-action pair
    by analyzing progress indicators in the accessibility tree and action types.
    """
    # 1. Immediate Success Detection
    # We search for outcome-based keywords in the next_state that indicate a goal was reached.
    success_indicators = [
        "success", "sent", "added", "created", "saved", 
        "done", "completed", "confirmed", "message sent",
        "event created", "todo added", "task completed",
        "successfully", "recorded", "updated"
    ]
    next_state_l = next_state.lower()
    if any(indicator in next_state_l for indicator in success_indicators):
        return 1.0
    
    # 2. Action-based Productivity Estimation
    # We analyze the action type and the resulting state change to estimate progress.
    
    # Handle 'fill' actions: e.g., fill('bid_1', 'Hello World')
    # Filling inputs is a fundamental step in almost all OpenApps tasks.
    fill_match = re.search(r"fill\s*\(\s*['\"]([^'\"]*)['\"]\s*,\s*['\"]([^'\"]*)['\"]\s*\)", action)
    if fill_match:
        input_text = fill_match.group(2)
        # If the text entered is now visible in the next state, it's a strong signal of successful input.
        if input_text and input_text in next_state:
            return 0.7
        # Even if not visible (e.g., buffered in a field), a 'fill' action is a productive step.
        return 0.4

    # Handle 'press' actions: e.g., press('bid_1', 'Enter')
    # Pressing keys, especially 'Enter', often triggers submission or confirmation.
    press_match = re.search(r"press\s*\(\s*['\"]([^'\"]*)['\"]\s*,\s*['\"]([^'\"]*)['\"]\s*\)", action)
    if press_match:
        key = press_match.group(2).lower()
        if key == 'enter':
            return 0.5
        return 0.2

    # Handle 'click' actions: e.g., click('bid_1')
    # Clicks drive navigation and interaction within the apps.
    click_match = re.search(r"click\s*\(\s*['\"]([^'\"]*)['\"]\s*\)", action)
    if click_match:
        # Use the number of 'bid' tags as a proxy for structural complexity/navigation.
        # A change in the number of interactive elements often means a page change or a modal opening.
        state_bids = len(re.findall(r"bid_\d+", state))
        next_bids = len(re.findall(r"bid_\d+", next_state))
        
        if state_bids != next_bids:
            return 0.5
        
        # If the number of bids is the same, check if the text content changed significantly.
        if len(state) > 0 and len(next_state) > 0:
            # Check for significant shifts in string length (potential navigation).
            if abs(len(next_state) - len(state)) / max(len(state), 1) > 0.2:
                return 0.4
            # Check for any text-based content change.
            if next_state.strip() != state.strip():
                return 0.3
        return 0.2

    # 3. Exploratory/Low-impact Actions
    # Scrolling is usually for discovery and does not directly progress a specific task goal.
    if "scroll" in action.lower():
        return 0.1
            
    # Default return for 'noop' or unrecognized action types.
    return 0.0