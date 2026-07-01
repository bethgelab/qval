import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value reflects the progress toward the goal based on key features
    extracted from the text representation of the state.
    """
    state_lower = state.lower()
    
    # 1. Terminal Success: The highest value is assigned when the task is completed.
    # ALFWorld typically indicates success with phrases like "task completed".
    if "task completed" in state_lower or "goal reached" in state_lower:
        return 1.0
    
    # 2. High Progress: Agent is holding the required object.
    # Holding an object is a major milestone in ALFWorld's "pick and place" tasks.
    if "you are holding" in state_lower or "carrying" in state_lower:
        # Increase value if the agent is also near common target locations/containers.
        targets = [
            "sink", "fridge", "counter", "table", "shelf", 
            "drawer", "cabinet", "bin", "microwave", "oven", "plate"
        ]
        if any(target in state_lower for target in targets):
            return 0.8
        return 0.6
    
    # 3. Medium Progress: Object discovery or state change (e.g., cleaning).
    # Cleaning an object is often a required intermediate step.
    if "cleaned" in state_lower or "washing" in state_lower:
        return 0.5
    
    # Seeing the target object indicates the agent has successfully navigated to it.
    if any(phrase in state_lower for phrase in ["you see", "is here", "visible"]):
        return 0.3
        
    # 4. Low Progress: Basic navigation.
    # Being in a room is a start, but doesn't guarantee proximity to the object.
    if "you are in" in state_lower or "moved to" in state_lower:
        return 0.1
        
    # Default value for states with no clear progress indicators.
    return 0.0