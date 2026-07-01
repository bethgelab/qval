import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the ALFWorld environment.
    The estimate is based on identifying progress towards common household tasks such as 
    navigation, object manipulation, and cleaning.
    """
    s_low = state.lower()
    a_low = action.lower()
    ns_low = next_state.lower()

    # 1. Immediate Success
    # If the next state indicates the task is complete, the Q-value is 1.0.
    success_keywords = ["success", "complete", "goal achieved", "task is complete", "you have finished"]
    if any(kw in ns_low for kw in success_keywords):
        return 1.0

    # 2. No change (invalid or redundant action)
    # If the state remains identical, the action did not result in any meaningful transition.
    if s_low == ns_low:
        return 0.0

    # Base value for any action that results in a state transition
    q_estimate = 0.1

    # 3. Navigation Progress
    # Matches: "go to kitchen", "move to the living room", "enter bedroom", "walk to the hallway"
    move_pattern = r"(?:go to|move to|walk to|enter) (?:a |the )?(\w+)"
    move_match = re.search(move_pattern, a_low)
    if move_match:
        target_room = move_match.group(1)
        if target_room in ns_low:
            # Check if the agent's location description actually updated to the target room
            if re.search(r"you are (in|now in) (?:a |the )?" + re.escape(target_room), ns_low):
                q_estimate += 0.4
            else:
                q_estimate += 0.2

    # 4. Object Manipulation: Picking up/Taking
    # Matches: "take apple", "pick up the apple", "get apple", "grab the apple"
    take_pattern = r"(?:take|pick up|get|grab) (?:a |the )?(\w+)"
    take_match = re.search(take_pattern, a_low)
    if take_match:
        obj = take_match.group(1)
        if obj in ns_low:
            # Check for possession-related descriptors in the next state
            if any(h in ns_low for h in ["hold", "have", "picked", "possession", "carrying", "you are holding"]):
                q_estimate += 0.5
            else:
                q_estimate += 0.3

    # 5. Object Manipulation: Placing/Putting
    # Matches: "put apple in fridge", "place the apple in the fridge", "drop apple in bin"
    put_pattern = r"(?:put|place|drop) (?:a |the )?(\w+) in (?:a |the )?(\w+)"
    put_match = re.search(put_pattern, a_low)
    if put_match:
        obj, loc = put_match.groups()
        if obj in ns_low and loc in ns_low:
            q_estimate += 0.5

    # 6. Object Manipulation: Cleaning/Using
    # Matches: "clean apple", "wash the apple", "scrub the table"
    clean_pattern = r"(?:clean|wash|scrub) (?:a |the )?(\w+)"
    clean_match = re.search(clean_pattern, a_low)
    if clean_match:
        obj = clean_match.group(1)
        if obj in ns_low and any(c in ns_low for c in ["clean", "washed", "shiny", "not dirty", "cleanly"]):
            q_estimate += 0.5

    # Cap the value at 0.99 to leave room for absolute success cases
    return min(q_estimate, 0.99)