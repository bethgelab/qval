import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment.
    The value is based on the agent's progress toward completing a household task,
    considering the typical pipeline: find object -> pick up -> clean -> move to target -> place.
    """
    state_lower = state.lower()
    
    # 1. Terminal Success State
    # Look for indicators that the task has been successfully completed.
    success_indicators = ["task completed", "success", "correctly placed", "goal reached"]
    if any(indicator in state_lower for indicator in success_indicators):
        return 1.0
    
    # 2. Progress Indicators
    # Common target locations in ALFWorld tasks
    targets = ["fridge", "microwave", "cabinet", "drawer", "table", "shelf", "sink"]
    is_at_target = any(t in state_lower for t in targets)
    is_holding = "holding" in state_lower
    is_clean = "clean" in state_lower
    is_dirty = "dirty" in state_lower
    
    # 3. State-Value Mapping based on progress stages
    # We use a tiered approach based on the likely stage of the episode.
    
    if is_holding:
        # Stage: Object is acquired.
        if is_clean:
            # Object is clean; agent just needs to reach target and place it.
            if is_at_target:
                return 0.9  # Ready to place
            else:
                return 0.7  # Holding clean object, moving to target
        elif is_dirty:
            # Object is dirty; agent needs to clean it first.
            if "sink" in state_lower or "sponge" in state_lower:
                return 0.6  # At cleaning station
            else:
                return 0.4  # Holding dirty object, moving to cleaning station
        else:
            # Object state unknown or neutral
            if is_at_target:
                return 0.6
            return 0.5
            
    else:
        # Stage: Object is not yet held.
        # Check if the object is visible in the current observation.
        # (Simplified: look for common objects or the word 'see')
        if "see" in state_lower:
            # Agent knows where the object is or is looking at it.
            if is_at_target:
                return 0.3  # At target location, but object not yet found/held
            return 0.3
        
        # Base case: Agent is searching or at the start.
        if is_at_target:
            return 0.2
        return 0.1