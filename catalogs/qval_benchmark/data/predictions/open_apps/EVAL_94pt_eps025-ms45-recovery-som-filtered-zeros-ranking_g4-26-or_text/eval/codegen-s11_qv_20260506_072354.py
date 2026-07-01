import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value (expected return) for a given action in a state.
    The Q-value represents the likelihood of completing the task successfully.
    """
    
    # 1. Immediate Reward Detection
    # In OpenApps, successful task completion often results in specific status messages.
    # Since the reward is binary (1.0 on success), if the next_state shows these, Q(s,a) is 1.0.
    success_indicators = [
        "message sent", "sent successfully", "event created", 
        "todo added", "task completed", "calendar event added",
        "search results", "found", "saved successfully",
        "message delivered", "item added", "navigation successful"
    ]
    
    next_state_lower = next_state.lower()
    if any(indicator in next_state_lower for indicator in success_indicators):
        return 1.0

    # 2. Action Parsing
    # Parse the action string (e.g., "click('12')", "fill('5', 'text')", "press('3', 'Enter')")
    # Using regex to extract the action type and its arguments.
    match = re.search(r"(\w+)\((.*)\)", action)
    if not match:
        return 0.0
        
    act_type = match.group(1)
    act_args = match.group(2)
    
    # 3. State Change Analysis
    # If the state hasn't changed after an action, it's likely a failed interaction.
    if next_state == state:
        if act_type in ('noop', 'scroll'):
            return 0.1  # Minor value for idling or scrolling if state is same
        else:
            return 0.05 # Low value for ineffective clicks/fills

    # 4. Progress Heuristics
    
    # Case A: Filling information
    # If the text provided in a 'fill' action is present in the next state, it's a strong indicator of progress.
    if act_type == 'fill':
        # The arguments are typically "'bid', 'text_value'"
        parts = act_args.split(',', 1)
        if len(parts) == 2:
            # Strip whitespace and surrounding quotes from the value
            val = parts[1].strip().strip("'\"")
            if val and val in next_state:
                # The value was successfully registered in the accessibility tree
                return 0.8
            else:
                # The value was typed but not yet reflected in the DOM/accessibility tree
                return 0.4
                
    # Case B: Interactions that drive workflows (Clicks and Presses)
    # Clicking buttons (like 'Submit', 'Send', 'Add') or pressing 'Enter' are high-utility actions.
    if act_type in ('click', 'press'):
        # A state change following a click or press is a good sign of progression.
        return 0.6
        
    # Case C: Scrolling
    # Scrolling is a navigation-assisting action but usually doesn't advance the goal directly.
    if act_type == 'scroll':
        return 0.2
        
    # Case D: General State Change
    # Any other action that results in a state change is treated as progress.
    return 0.3