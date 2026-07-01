import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an ALFWorld environment.
    The value is an approximation of the probability of reaching the goal.
    """
    s = state.lower()
    
    # 1. Terminal Success Check
    # If the environment indicates success, the value is 1.0.
    if "success" in s or "goal achieved" in s or "task completed" in s:
        return 1.0
        
    # 2. Task and Observation Extraction
    task = ""
    obs = s
    # ALFWorld states often contain "Task: [task] Observation: [obs]"
    if "task:" in s:
        parts = re.split(r'task:|observation:', s)
        if len(parts) >= 3:
            task = parts[1].strip()
            obs = parts[2].strip()
        elif len(parts) == 2:
            # If 'observation:' is missing, the second part is likely the task
            task = parts[1].strip()
    
    # 3. Goal Requirement Parsing
    # We attempt to identify the target object, target location, and required state.
    target_obj = None
    target_loc = None
    target_state = None
    
    if task:
        # Case A: "put [obj] in [loc]" or "move [obj] to [loc]"
        # Regex looks for an object followed by a preposition and a location
        put_match = re.search(r'(?:put|move|place|go|put\s+the|move\s+the)\s+(?:a\s+|the\s+|an\s+)?(\w+)\s+(?:in|on|to|into|at|inside)\s+(?:a\s+|the\s+|an\s+)?(\w+)', task)
        if put_match:
            target_obj, target_loc = put_match.groups()
            # If the task mentions 'clean', we assume a 'clean' state is required
            if "clean" in task:
                target_state = "clean"
        else:
            # Case B: "clean [obj]"
            clean_match = re.search(r'(?:clean|wash|scrub)\s+(?:a\s+|the\s+|an\s+)?(\w+)', task)
            if clean_match:
                target_obj = clean_match.group(1)
                target_state = "clean"
            else:
                # Case C: "find [obj]"
                find_match = re.search(r'(?:find|locate|search\s+for)\s+(?:a\s+|the\s+|an\s+)?(\w+)', task)
                if find_match:
                    target_obj = find_match.group(1)
                else:
                    # Fallback: extract potential nouns by ignoring common verbs/prepositions
                    words = re.findall(r'\w+', task)
                    ignore = {'put', 'move', 'place', 'go', 'the', 'a', 'an', 'in', 'on', 'to', 'at', 'into', 'clean', 'find', 'task', 'is'}
                    for w in words:
                        if w not in ignore:
                            if not target_obj:
                                target_obj = w
                            elif not target_loc:
                                target_loc = w
                            break

    # 4. State-Value Estimation
    # If no goal can be parsed, return a baseline.
    if not target_obj:
        return 0.1

    # Check for presence of components in the current observation
    obj_in_obs = target_obj in obs
    loc_in_obs = target_loc in obs if target_loc else False
    state_satisfied = True
    if target_state and target_state not in obs:
        state_satisfied = False
    
    # Score calculation based on proximity and presence
    if target_loc:
        # If it's a "put/move" task
        if obj_in_obs and loc_in_obs:
            # Check if the object is already at/in the location via proximity regex
            # e.g., "apple...in...fridge"
            proximity_pattern = rf"{target_obj}.*?\b(?:in|on|at|inside|into|next\s+to)\b.*?{target_loc}"
            if re.search(proximity_pattern, obs) and state_satisfied:
                return 0.95
            else:
                # Both are present, but either they aren't together or state isn't met
                return 0.5 if state_satisfied else 0.3
        elif obj_in_obs:
            return 0.4 if state_satisfied else 0.2
        elif loc_in_obs:
            return 0.2
        else:
            return 0.1
            
    elif target_state:
        # If it's a "clean" task
        if obj_in_obs:
            return 0.9 if state_satisfied else 0.4
        else:
            return 0.1
            
    else:
        # If it's a "find" task or generic
        if obj_in_obs:
            return 0.5
        else:
            return 0.1

    return 0.1