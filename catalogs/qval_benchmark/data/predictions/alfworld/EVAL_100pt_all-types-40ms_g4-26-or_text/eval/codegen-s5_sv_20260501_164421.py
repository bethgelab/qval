import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for an ALFWorld state by parsing the instruction 
    and the current observation to determine how close the agent is to the goal.
    """
    s = state.lower()
    
    # 1. Check for immediate success
    if any(term in s for term in ["task completed", "goal reached", "success", "finished"]):
        return 1.0

    # 2. Split instruction and observation
    # ALFWorld typically presents the instruction first, followed by 'Observation:'
    instr = ""
    obs = s
    if "observation:" in s:
        parts = s.split("observation:", 1)
        instr = parts[0].strip()
        obs = parts[1].strip()
    else:
        # Heuristic: instruction is usually the first sentence or before a period
        dot_idx = s.find('.')
        if dot_idx != -1:
            instr = s[:dot_idx].strip()
            obs = s[dot_idx + 1:].strip()
        else:
            instr = s

    if not instr:
        instr = s

    # 3. Parse the instruction for key task components: object, location, and type
    # Extract target object (the noun following a command verb)
    obj = None
    obj_match = re.search(r"(?:put|pick|clean|get|move|place|wash|find|take|grab|drop)\s+(?:the\s+|a\s+|an\s+)?(\w+)", instr)
    if obj_match:
        obj = obj_match.group(1)
    else:
        # Fallback: find the first noun following a common article
        alt_obj = re.search(r"(?:the|a|an)\s+(\w+)", instr)
        if alt_obj:
            obj = alt_obj.group(1)
    
    if not obj:
        return 0.5  # Neutral value if we can't identify what to do

    # Extract target location (the noun following a preposition)
    loc = None
    loc_match = re.search(r"(?:in|on|at|inside|to|into)\s+(?:the\s+|a\s+|an\s+)?(\w+)", instr)
    if loc_match:
        loc_cand = loc_match.group(1)
        # Ensure the location isn't accidentally the object itself
        if loc_cand != obj:
            loc = loc_cand
    
    # Identify task type
    is_cleaning = "clean" in instr or "wash" in instr

    # 4. Evaluate progress based on the observation
    
    # Case A: Placement/Movement Task (Object must be at Location)
    if loc:
        # Check if object is at the target location in the observation
        # Matches "apple in the fridge" or "fridge contains the apple"
        pattern_obj_at_loc = rf"\b{obj}\b.*?\s+(?:in|on|at|inside|into)\s+(?:the\s+|a\s+|an\s+)?\b{loc}\b"
        pattern_loc_has_obj = rf"\b{loc}\b.*?\s+(?:contains|has|there is a|there's a).*?\b{obj}\b"
        
        if re.search(pattern_obj_at_loc, obs) or re.search(pattern_loc_has_obj, obs):
            return 1.0
        
        # Check if the agent is holding the object
        holding_pattern = rf"(?:holding|have|with|carrying|got)\s+(?:the\s+|a\s+|an\s+)?\b{obj}\b"
        if re.search(holding_pattern, obs):
            return 0.7
        
        # Check if the object is visible/present in the current room
        if re.search(rf"\b{obj}\b", obs):
            return 0.4
            
        return 0.1

    # Case B: Cleaning Task (Object must be 'clean')
    elif is_cleaning:
        if re.search(rf"\b{obj}\b.*?\s+is\s+clean", obs):
            return 1.0
        if re.search(rf"\b{obj}\b.*?\s+is\s+(?:dirty|dirty|dirty)", obs):
            return 0.2
        if re.search(rf"\b{obj}\b", obs):
            return 0.5
        return 0.1

    # Case C: Simple Pick-up Task (No location specified)
    else:
        holding_pattern = rf"(?:holding|have|with|carrying|got)\s+(?:the\s+|a\s+|an\s+)?\b{obj}\b"
        if re.search(holding_pattern, obs):
            return 1.0
        if re.search(rf"\b{obj}\b", obs):
            return 0.5
        return 0.1

    return 0.1