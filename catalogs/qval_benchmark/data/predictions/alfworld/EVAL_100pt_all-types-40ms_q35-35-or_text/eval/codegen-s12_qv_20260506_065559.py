import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Normalize inputs for case-insensitive matching
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()

    # 1. Success Detection
    # High Q-value if the episode is successfully completed
    success_patterns = ["success", "complete", "congratulations", "task complete", "correct"]
    for pattern in success_patterns:
        if pattern in next_state_lower:
            return 1.0

    # 2. Failure Detection
    # Low Q-value if the action failed or state is invalid
    fail_patterns = ["fail", "cannot", "invalid", "error", "cannot do that", "try again"]
    for pattern in fail_patterns:
        if pattern in next_state_lower:
            return 0.0

    # 3. Stagnation Detection
    # No progress if state hasn't changed
    if state == next_state:
        return 0.0

    # 4. Progress Estimation
    # Base score for taking any action
    score = 0.1

    # Heuristic: Location Change
    # Look for "You are in the [room]" pattern
    loc_match = re.search(r"you are in the (\w+)", next_state_lower)
    state_loc_match = re.search(r"you are in the (\w+)", state_lower)
    
    if loc_match and state_loc_match:
        if loc_match.group(1) != state_loc_match.group(1):
            score += 0.3
    else:
        # Fallback: check for known room/location keywords
        locations = ["kitchen", "living room", "bathroom", "bedroom", "countertop", "sink", "fridge", "microwave", "cabinet"]
        state_locs = [l for l in locations if l in state_lower]
        next_locs = [l for l in locations if l in next_state_lower]
        if set(next_locs) != set(state_locs):
            score += 0.2

    # Heuristic: Object Manipulation
    # Count prepositions indicating placement (in, on, at)
    placement_pattern = r"\b(in|on|at)\b"
    state_placements = len(re.findall(placement_pattern, state_lower))
    next_placements = len(re.findall(placement_pattern, next_state_lower))
    
    if next_placements > state_placements:
        score += 0.3

    # Heuristic: Action Consistency
    # If action is navigation, ensure location changed
    if "go to" in action_lower:
        if (loc_match and state_loc_match and loc_match.group(1) != state_loc_match.group(1)) or \
           (set(next_locs) != set(state_locs)):
            score += 0.1

    # Clamp score to [0, 1]
    if score > 1.0:
        score = 1.0
    if score < 0.0:
        score = 0.0

    return score