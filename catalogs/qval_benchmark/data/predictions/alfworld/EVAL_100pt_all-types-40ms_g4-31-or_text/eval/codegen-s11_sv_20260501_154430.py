import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment state.
    The value is based on the agent's progress toward the goal:
    Discovery -> Possession -> Processing -> Target Proximity -> Completion.
    """
    if not state:
        return 0.0
        
    s = state.lower()
    
    # 1. Terminal Success
    # Look for explicit signs that the task has been completed.
    success_markers = ["task completed", "successfully put", "successfully placed", "episode finished"]
    if any(marker in s for marker in success_markers):
        return 1.0
    
    # 2. Identify key state features
    # Possession: Is the agent holding the target object?
    is_holding = "holding" in s or "picked up" in s
    
    # Target Proximity: Is the agent near a common target location?
    target_locations = [
        "fridge", "sink", "table", "counter", "drawer", 
        "cabinet", "bin", "shelf", "microwave", "oven"
    ]
    is_near_target = any(loc in s for loc in target_locations)
    
    # Progress/Processing: Has the agent performed a necessary intermediate action?
    progress_markers = ["cleaned", "opened", "examined"]
    has_progress = any(marker in s for marker in progress_markers)
    
    # Discovery: Has the agent located the target object in the environment?
    discovery_markers = ["there is a", "you see a", "is on the"]
    has_found_object = any(marker in s for marker in discovery_markers)
    
    # 3. Value Calculation based on progress hierarchy
    # Priority 1: Possession + Target Proximity (Very close to success)
    if is_holding and is_near_target:
        return 0.9
    
    # Priority 2: Possession + Processing (Ready to be placed)
    if is_holding and has_progress:
        return 0.8
    
    # Priority 3: Possession (Object secured, but needs processing or navigation)
    if is_holding:
        return 0.7
    
    # Priority 4: Processing performed (e.g., cleaned the object before picking it up, or opened a door)
    if has_progress:
        return 0.5
    
    # Priority 5: Discovery (Object found but not yet possessed)
    if has_found_object:
        return 0.3
    
    # Priority 6: Target Proximity (Agent is in the right place, but object is not yet held)
    if is_near_target:
        return 0.2
    
    # Baseline: Agent is active in the environment but has made no significant progress
    return 0.1