import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for an action in ALFWorld by analyzing the progress 
    made toward the stated goal in the text-based environment.
    """
    ns_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()

    # 1. Check for immediate task completion in the next state
    if "task complete" in ns_lower or "successfully" in ns_lower or "you have finished" in ns_lower:
        return 1.0

    # 2. Parse the goal from the current state
    # Standard ALFWorld format: "Your task is to [action] [object] in [location]."
    obj = None
    loc = None
    goal_match = re.search(r"task is to (.*?)\.", state_lower)
    
    if goal_match:
        goal_text = goal_match.group(1)
        # Split the goal into action/object part and location part
        # Common prepositions for movement/placement in ALFWorld
        parts = re.split(r"\s+(?:in|to|at|on|into)\s+", goal_text, maxsplit=1)
        
        if len(parts) == 2:
            verb_obj_part = parts[0]
            loc_part = parts[1]
            
            # Extract location (remove "the" if present)
            loc = loc_part.replace("the ", "").strip()
            
            # Extract object (look for word(s) after 'the', 'a', or 'an')
            obj_match = re.search(r"(?:the|a|an)\s+(.*)", verb_obj_part)
            if obj_match:
                obj = obj_match.group(1).strip()
            else:
                # Fallback to the last word of the verb_obj_part
                obj = verb_obj_part.split()[-1]
        else:
            # Goal might be simpler, like "clean the plate" (no location part)
            verb_obj_part = goal_text
            obj_match = re.search(r"(?:the|a|an)\s+(.*)", verb_obj_part)
            if obj_match:
                obj = obj_match.group(1).strip()
            else:
                obj = verb_obj_part.split()[-1]

    # 3. Evaluate progress based on parsed goal components
    score = 0.0
    
    if obj and loc:
        # Check for high-level goal fulfillment (object in location)
        # Matches: "the apple is in the fridge", "apple is in fridge", etc.
        if obj in ns_lower and loc in ns_lower:
            if any(p in ns_lower for p in [" is in ", " is at ", " is on ", " is into "]):
                score = 1.0
            else:
                score = 0.3 # Just seeing both in the room
        
        # Check if the agent is holding the target object
        if obj in ns_lower and any(h in ns_lower for h in ["holding", "have", "got", "hand"]):
            if score < 0.7:
                score = 0.7
                
        # Check if the agent is in the target location
        if loc in ns_lower and any(p in ns_lower for p in ["you are in", "you are at", "you are on"]):
            if score < 0.4:
                score = 0.4
        
        # If the object is visible but not held/placed
        if obj in ns_lower and score < 0.3:
            score = 0.3

    elif obj:
        # Check for completion of single-object tasks (e.g., cleaning)
        if obj in ns_lower and any(c in ns_lower for c in ["is clean", "is cleaned", "is spotless"]):
            score = 1.0
        elif obj in ns_lower and any(h in ns_lower for h in ["holding", "have", "got", "hand"]):
            score = 0.7
        elif obj in ns_lower:
            score = 0.4

    elif loc:
        # Check if agent has arrived at a target location
        if loc in ns_lower and any(p in ns_lower for p in ["you are in", "you are at", "you are on"]):
            score = 0.4

    # 4. Handle cases where no specific goal components were parsed or matched
    if score == 0.0:
        # Check if the action itself is relevant to the potential goal
        if obj and obj in action_lower:
            score = 0.2
        elif loc and loc in action_lower:
            score = 0.2
        else:
            # Baseline probability of success for a generic step
            score = 0.05

    # Ensure Q-value is within [0, 1]
    return max(0.0, min(1.0, score))