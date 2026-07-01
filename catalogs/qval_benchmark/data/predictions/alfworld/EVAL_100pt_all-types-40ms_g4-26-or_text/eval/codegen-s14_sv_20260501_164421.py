import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given state in the ALFWorld environment.
    The value reflects the progress toward completing the task.
    """
    state_lower = state.lower()
    
    # 1. Extract Instruction
    # The instruction might be prefixed with "Instruction:" or be the first part of the text.
    instruction = ""
    instr_match = re.search(r"instruction:\s*(.*?)(?:\.|\n|$)", state_lower)
    if instr_match:
        instruction = instr_match.group(1)
    else:
        # Fallback: Try to find the first sentence before the description of the room.
        first_sentence_match = re.search(r"^(.*?)(?:\.|\n|you are in)", state_lower)
        if first_sentence_match:
            instruction = first_sentence_match.group(1).strip()
        else:
            instruction = state_lower

    # 2. Parse Instruction for target object and target location
    target_obj = ""
    target_loc = ""
    
    # Patterns: "put the [obj] in the [loc]", "place [obj] in [loc]", "move [obj] to [loc]"
    m = re.search(r"(?:put|place|move)\s+(?:the\s+)?(.*?)\s+(?:in|to)\s+(?:the\s+)?(.*)", instruction)
    if m:
        target_obj = m.group(1).strip()
        target_loc = m.group(2).strip()
    else:
        # Patterns: "find the [obj]", "get [obj]", "clean [obj]", "wash [obj]"
        m = re.search(r"(?:find|get|clean|wash)\s+(?:the\s+)?(.*)", instruction)
        if m:
            target_obj = m.group(1).strip()
            
    # Clean up trailing punctuation from parsed targets
    target_obj = re.sub(r'[^\w\s]', '', target_obj)
    target_loc = re.sub(r'[^\w\s]', '', target_loc)

    if not target_obj:
        return 0.0

    # 3. Parse Observation
    # Find inventory: "You are holding a/an [item]."
    inventory = []
    inv_match = re.search(r"you are holding\s+(?:a|an)?\s*(.*?)(?:\.|\n|$)", state_lower)
    if inv_match:
        inv_str = inv_match.group(1).strip()
        if inv_str and inv_str != "nothing":
            inventory = [inv_str]

    # Find visible objects: "There is a/an [item], [item], and [item]."
    visible_objs = []
    objs_match = re.search(r"there is (.*?)\.", state_lower)
    if objs_match:
        objs_str = objs_match.group(1)
        # Split by comma or 'and'
        parts = re.split(r',|and', objs_str)
        for p in parts:
            p = p.strip()
            # Remove leading 'a ' or 'an '
            p = re.sub(r'^(?:a|an)\s+', '', p)
            if p:
                visible_objs.append(p)

    # 4. Evaluate Progress
    # Rule 1: Check if goal is already met (e.g., "the apple is in the fridge")
    if target_obj and target_loc:
        # Use a loose regex to check if target object is in target location
        completion_pattern = rf"{target_obj}.*?is.*?in.*?{target_loc}"
        if re.search(completion_pattern, state_lower):
            return 1.0
    
    # Rule 2: Check if the agent is holding the target object
    is_held = any(target_obj in item for item in inventory)
    if is_held:
        # If we are holding the object and the target location is visible in the current room
        if target_loc and any(target_loc in item for item in visible_objs):
            return 0.9
        return 0.7
    
    # Rule 3: Check if the target object is visible in the current room
    if any(target_obj in item for item in visible_objs):
        return 0.5
            
    # Rule 4: Check if the target location (if it's an object) is visible in the current room
    if target_loc and any(target_loc in item for item in visible_objs):
        return 0.4
            
    # Rule 5: Check if the target object or location is mentioned anywhere in the observation
    if target_obj in state_lower or (target_loc and target_loc in state_lower):
        return 0.2
            
    return 0.0