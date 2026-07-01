import re
from collections import Counter

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment based on 
    heuristic features extracted from the state string.
    
    Heuristics:
    1. Proximity to goal: Check if goal-related phrases are mentioned in the observation.
    2. Inventory: Having the target object in inventory is a strong positive signal.
    3. Obstacles: Presence of "closed", "dirty", or "broken" states usually requires extra steps.
    4. Location: Being in the correct room or near the target location increases value.
    5. Efficiency: Fewer steps remaining (implied by progress) increases value.
    """
    
    # Normalize state for analysis
    state_lower = state.lower()
    
    # Initialize base value
    value = 0.0
    
    # 1. Check for successful completion (Goal achieved)
    # Common patterns in ALFWorld success messages
    success_patterns = [
        r"task\s+complete",
        r"success",
        r"congratulations",
        r"done",
        r"placed\s+in\s+the",
        r"moved\s+to"
    ]
    for pattern in success_patterns:
        if re.search(pattern, state_lower):
            return 1.0  # Maximum value if goal is reached
    
    # 2. Check if the agent is holding the target object
    # Patterns: "holding", "in your inventory", "picked up"
    holding_patterns = [
        r"you are holding",
        r"in your inventory",
        r"picked up",
        r"carrying"
    ]
    is_holding = any(re.search(p, state_lower) for p in holding_patterns)
    if is_holding:
        value += 0.3
    
    # 3. Check if the object is already in the target location
    # Patterns: "already in", "placed in", "is on", "is in" (context dependent)
    # We look for phrases indicating the object is where it needs to be
    target_location_patterns = [
        r"already in",
        r"is on the",
        r"is in the",
        r"placed on the",
        r"placed in the"
    ]
    # Heuristic: If the state mentions the object is in a container/room that matches typical goals
    # This is a rough approximation based on common goal structures (e.g., "put coffee mug in microwave")
    # We assume if the text says "is on the table" or similar for the object, it might be close.
    # However, without explicit goal definition in the state string, we look for "already"
    if "already" in state_lower:
        value += 0.4
    elif re.search(r"placed on the", state_lower) or re.search(r"placed in the", state_lower):
        value += 0.3
    
    # 4. Penalize for obstacles (extra steps needed)
    # "closed", "dirty", "dirty", "unplug", "turn off" imply sub-tasks
    obstacle_patterns = [
        r"closed",
        r"dirty",
        r"unplugged",
        r"turned off",
        r"broken"
    ]
    obstacle_count = sum(1 for p in obstacle_patterns if p in state_lower)
    value -= 0.1 * obstacle_count
    
    # 5. Check for navigation context
    # If the state mentions "You are in" and the room name, it's a valid state.
    # If the state mentions "go to", it implies navigation is needed.
    if "go to" in state_lower:
        value -= 0.1  # Slight penalty for needing to move
    
    # 6. Check for "look at" or "examine" context
    # Sometimes the state is just a description. If it's a description of the target object, good.
    # If it's a description of the room, neutral.
    # We rely on the "holding" and "already" checks mostly.
    
    # 7. Step efficiency heuristic (approximate)
    # If the state is very verbose about the room contents, it might be early in the episode.
    # If it's short and focused on the object, it might be late.
    # This is a weak signal.
    word_count = len(state.split())
    if word_count > 200:
        value -= 0.05 # Very long state might mean complex navigation or early exploration
    
    # Clamp value between 0 and 1
    return max(0.0, min(1.0, value))