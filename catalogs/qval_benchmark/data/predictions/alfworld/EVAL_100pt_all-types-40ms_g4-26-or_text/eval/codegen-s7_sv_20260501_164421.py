import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given state in the ALFWorld environment.
    The value is an approximation based on task progress and object visibility.
    """
    state_lower = state.lower()
    
    # 1. Immediate success check
    # If the environment indicates the task is finished.
    success_indicators = ["completed", "finished", "success", "task is done", "you have completed", "is finished"]
    if any(ind in state_lower for ind in success_indicators):
        return 1.0
        
    # 2. Separate Instruction and Observation
    # The state typically contains an 'Instruction' and an 'Observation'.
    instr = ""
    obs = state_lower
    
    # Try to find the instruction/goal in the text
    instr_match = re.search(r"(?:instruction|goal|task):\s*(.*?)(?:\.|observation:|$)", state_lower)
    if instr_match:
        instr = instr_match.group(1).strip()
        # Look for the observation part specifically
        obs_match = re.search(r"observation:\s*(.*)", state_lower, re.DOTALL)
        if obs_match:
            obs = obs_match.group(1).strip()
    else:
        # Fallback if no explicit "Instruction:" label exists
        # If the text is short, it might just be the instruction.
        if len(state_lower.split()) < 15:
            instr = state_lower
        else:
            return 0.1

    if not instr:
        return 0.1

    # 3. Parsing Instruction for Action, Subject, and Target
    # Common verbs used in ALFWorld tasks
    actions = ["put", "clean", "pick up", "find", "move", "get", "drop", "place", "wash", "take"]
    target_action = None
    for a in actions:
        if a in instr:
            target_action = a
            break
    
    # Extract nouns using common patterns (the [noun], a [noun], an [noun])
    # This is an effective heuristic for ALFWorld instructions.
    nouns = re.findall(r"(?:the|a|an)\s+([a-z]+)", instr)
    
    # In most ALFWorld tasks:
    # "Put the apple in the fridge" -> subject='apple', target='fridge'
    # "Clean the apple" -> subject='apple', target=None
    # "Pick up the apple" -> subject='apple', target=None
    subject = nouns[0] if len(nouns) > 0 else None
    target = nouns[1] if len(nouns) > 1 else None
    
    if not subject:
        return 0.1

    # 4. Evaluating Progress in the Observation
    # Check if the subject is mentioned in the current observation.
    subject_in_obs = subject in obs
    
    if not subject_in_obs:
        # If the subject isn't visible, the agent might be in the wrong room.
        return 0.1

    # Base score for finding the subject
    score = 0.4
    
    # Check if the subject is currently being held/carried.
    # This is high progress for "pick up" or "move" tasks.
    is_held = any(word in obs for word in ["holding", "in your hand", "carrying", "you have the", "you are holding"])
    if is_held:
        score = 0.6
            
    # Check for task completion via "at target" (for put/move/place tasks)
    if target and target in obs:
        # Regex to detect if the subject is at the target location (e.g., "apple in the fridge")
        # We look for [subject] followed by a preposition and then [target].
        pattern = rf"{subject}.*?(?:in|on|at|inside|on top of|placed in|placed on)\s+(?:the\s+)?{target}"
        if re.search(pattern, obs):
            score = 1.0
            
    # Check for task completion via "cleanliness" (for clean/wash tasks)
    if target_action in ["clean", "wash"]:
        if "clean" in obs or "is clean" in obs or "is washed" in obs:
            score = 1.0
        elif "dirty" in obs:
            # If the subject is visible but specifically noted as dirty
            score = 0.4
                
    # Explicitly re-verify "put" tasks if we haven't hit 1.0 yet
    if target_action in ["put", "place", "move"] and target:
        if target in obs:
            pattern = rf"{subject}.*?(?:in|on|at|inside|on top of)\s+(?:the\s+)?{target}"
            if re.search(pattern, obs):
                score = 1.0

    # Final clamping to ensure the value is in [0.0, 1.0]
    if score > 1.0: score = 1.0
    if score < 0.0: score = 0.0
    
    return float(score)