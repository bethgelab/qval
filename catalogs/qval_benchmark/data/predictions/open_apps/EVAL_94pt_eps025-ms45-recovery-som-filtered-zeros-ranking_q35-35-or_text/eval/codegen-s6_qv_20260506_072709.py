import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    
    The estimation is based on heuristic features derived from the text representation of the
    accessibility tree, the action taken, and the change in state.
    
    Heuristics:
    1. Goal Achievement: If the next_state explicitly indicates the goal is reached (e.g., "Task Complete", "Success", "Done"), return 1.0.
    2. Action Validity: If the action is a 'noop' or 'scroll' without significant state change, the Q-value is low (penalty for inefficiency).
    3. Progress: If the next_state contains keywords related to the specific app goals (e.g., "Event added", "Message sent", "Todo created") compared to the state, reward positively.
    4. Error States: If the next_state indicates an error (e.g., "Error", "Invalid", "Not found"), return a negative value.
    5. Efficiency: If the action moves closer to a target element (e.g., clicking a button that is now disabled or changed), reward.
    6. Baseline: Default to a small negative value to penalize non-progressive steps.
    """
    
    # Normalize strings for analysis
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # 1. Check for immediate goal completion in the next state
    goal_indicators = ["task complete", "success", "done", "goal achieved", "event added", "message sent", "todo created", "event created", "message sent successfully"]
    for indicator in goal_indicators:
        if indicator in next_state_lower:
            return 1.0
            
    # 2. Check for error states
    error_indicators = ["error", "invalid", "not found", "failed", "unable to", "cannot", "exception"]
    for indicator in error_indicators:
        if indicator in next_state_lower:
            return -1.0
            
    # 3. Analyze action type for efficiency penalties
    if action_lower.startswith("noop"):
        # No action taken, likely a penalty unless already done (handled above)
        return -0.1
        
    if action_lower.startswith("scroll"):
        # Scrolling is usually a preparatory step, slight penalty unless it reveals a key element
        # We check if the next state has new interactive elements or specific targets
        if "click" in next_state_lower or "button" in next_state_lower:
            return -0.05
        return -0.05
        
    # 4. Analyze progress based on action type
    # Clicking a button that leads to a form or confirmation is good progress
    if action_lower.startswith("click"):
        # Check if the next state reflects a state change typical of a successful click
        # e.g., a modal appears, a form loads, a button becomes disabled
        if "modal" in next_state_lower or "form" in next_state_lower or "dialog" in next_state_lower:
            return 0.1
        if "disabled" in next_state_lower or "loading" in next_state_lower:
            return 0.05
        # Generic click on a known target (e.g., specific app names)
        if "todo" in next_state_lower and "add" in action_lower:
            return 0.2
        if "calendar" in next_state_lower and "event" in action_lower:
            return 0.2
        if "messenger" in next_state_lower and "message" in action_lower:
            return 0.2
            
    # 5. Fill action analysis
    if action_lower.startswith("fill"):
        # Filling a field is progress, but usually needs a submit
        if "submit" in next_state_lower or "send" in next_state_lower:
            return 0.3
        return 0.1
        
    # 6. Default penalty for non-specific actions or lack of clear progress
    # This encourages the agent to find the shortest path
    return -0.05