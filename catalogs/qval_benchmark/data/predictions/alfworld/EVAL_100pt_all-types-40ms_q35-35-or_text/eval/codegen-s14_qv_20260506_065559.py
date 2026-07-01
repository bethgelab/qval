import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Normalize inputs for analysis
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # 1. Check for Terminal States
    # Success indicators
    success_keywords = ["success", "goal reached", "complete", "done", "task complete"]
    for kw in success_keywords:
        if kw in next_state_lower:
            return 1.0
            
    # Failure indicators
    failure_keywords = ["fail", "error", "invalid", "cannot", "cannot find"]
    for kw in failure_keywords:
        if kw in next_state_lower:
            return 0.0
            
    # 2. Calculate Progress Score
    score = 0.0
    
    # Check if state actually changed (basic heuristic)
    # If strings are identical, no progress was made (except maybe wait, but rare)
    if state == next_state:
        score -= 0.2
    else:
        # Positive baseline for valid transition
        score += 0.1
        
    # 3. Analyze Action-Outcome Consistency
    # Manipulation Actions
    if "put" in action_lower:
        # Look for object placement in next_state
        if "on" in next_state_lower or "in" in next_state_lower:
            score += 0.3
        elif "put" in next_state_lower: # Action repeated in description might mean failure or just description
             # If action is repeated in state, it might be a failed command logged
             pass 
        else:
            score -= 0.1 # Expected placement didn't happen
            
    elif "take" in action_lower or "pick" in action_lower:
        # Look for inventory/hand indicators
        if "hand" in next_state_lower or "holding" in next_state_lower or "picked up" in next_state_lower:
            score += 0.3
        else:
            score -= 0.1
            
    elif "clean" in action_lower or "wash" in action_lower:
        # Look for cleanliness state
        if "clean" in next_state_lower:
            score += 0.4
        else:
            score -= 0.1
            
    elif "turn on" in action_lower:
        # Look for power state
        if "on" in next_state_lower:
            score += 0.3
        elif "off" in next_state_lower:
            score -= 0.2 # Turned off instead of on
        else:
            score -= 0.1
            
    elif "turn off" in action_lower:
        if "off" in next_state_lower:
            score += 0.3
        else:
            score -= 0.1
            
    # Navigation Actions
    elif "go to" in action_lower or "go" in action_lower:
        # Check for room change
        # Extract rooms from state and next_state roughly
        state_rooms = re.findall(r"in the ([\w\s]+)", state_lower)
        next_rooms = re.findall(r"in the ([\w\s]+)", next_state_lower)
        
        # If room list changed, progress made
        if set(state_rooms) != set(next_rooms):
            score += 0.2
        else:
            # If no room change, maybe still valid (e.g. moved within room)
            # But navigation usually implies room change in ALFWorld
            pass
            
    # 4. Clamp and Return
    # Ensure Q-value is within [0.0, 1.0]
    return max(0.0, min(1.0, score))