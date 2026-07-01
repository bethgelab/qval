import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    The Q-value is estimated based on the value of the resulting state (V(s')),
    distinguishing between progress towards goals (holding objects, placing objects,
    navigating) and failure.
    """
    ns = next_state.lower()
    
    # 1. Check for immediate success/completion
    # ALFWorld typically signals completion with phrases like "task completed" 
    # or "successfully completed".
    success_indicators = ["task completed", "goal reached", "successfully completed", "successfully achieved"]
    if any(indicator in ns for indicator in success_indicators):
        return 1.0
        
    # 2. Check for invalid or no-op actions
    # If the environment explicitly says nothing changed or the action was invalid,
    # the expected return is effectively 0.
    failure_indicators = ["nothing changed", "you can't", "not possible", "is not possible", "invalid command"]
    if any(indicator in ns for indicator in failure_indicators):
        return 0.0
        
    # 3. Feature Extraction to estimate V(s')
    # We look for specific structural changes in the text:
    # - Placement: "sponge is in the sink"
    # - Holding: "holding a sponge"
    # - Room: "you are in the kitchen"
    
    # Placement detection: looks for "[object] is in [location]"
    # We filter out "you are in" by checking if the first word is 'you'.
    placements = set()
    placement_matches = re.findall(r"(\w+)\s+(?:is|are)\s+(?:in|on|at)\s+(?:the\s+)?(\w+)", ns)
    for obj, loc in placement_matches:
        if obj != "you":
            placements.add((obj, loc))
            
    # Holding detection: looks for "holding [object]"
    holding = None
    holding_match = re.search(r"holding\s+(?:a|an|the\s+)?(\w+)", ns)
    if holding_match:
        holding = holding_match.group(1)
        
    # Room detection: looks for "you are in [room]"
    room = None
    room_match = re.search(r"you\s+are\s+in\s+(?:the\s+)?(\w+)", ns)
    if room_match:
        room = room_match.group(1)
        
    # 4. Value Estimation Logic
    # We assign higher values to states that represent more advanced task stages.
    # - Placement (near completion) -> 0.9
    # - Holding (intermediate) -> 0.5
    # - Room (initial/navigation) -> 0.2
    # - Else -> 0.0
    
    v_s_prime = 0.0
    if placements:
        v_s_prime = 0.9
    elif holding:
        v_s_prime = 0.5
    elif room:
        v_s_prime = 0.2
    else:
        v_s_prime = 0.0
        
    return float(v_s_prime)