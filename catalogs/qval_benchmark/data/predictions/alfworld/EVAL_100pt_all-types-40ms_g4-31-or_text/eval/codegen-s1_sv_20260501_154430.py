import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is based on the agent's progress towards the goal:
    - Exploring/Navigating (Low Value)
    - Locating Object (Low-Medium Value)
    - Holding Object (Medium-High Value)
    - Placing Object (High Value)
    - Task Completed (Maximum Value)
    """
    # Normalize state to lowercase for consistent matching
    s = state.lower()
    
    # 1. Terminal Success: The task is explicitly completed.
    if "task completed" in s or ("success" in s and "finished" in s):
        return 1.0
    
    # 2. High Value: The final action (putting/placing) has likely occurred.
    # "successfully put the apple in the fridge" or "the apple is now in the fridge"
    if "successfully put" in s or "successfully placed" in s or "is now in" in s or "is now on" in s:
        return 0.9
    
    # 3. Medium-High Value: The agent is holding the target object.
    # This is the most significant milestone before completion.
    if "holding" in s:
        return 0.7
    
    # 4. Medium Value: The agent has successfully picked up the object.
    if "picked up" in s:
        return 0.6
    
    # 5. Low-Medium Value: The target object has been located in the environment.
    # Common descriptors in ALFWorld include "the [object] is on the [surface]".
    if "is on the" in s or "is in the" in s:
        return 0.4
    
    # 6. Low Value: The agent is in a room, potentially searching or navigating.
    if "you are in" in s:
        return 0.2
    
    # Default value for the initial or uninformative states.
    return 0.0