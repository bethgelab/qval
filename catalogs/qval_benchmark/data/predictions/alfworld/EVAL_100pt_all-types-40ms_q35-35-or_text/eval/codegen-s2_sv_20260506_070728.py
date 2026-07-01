import re

def signal_function(state: str) -> float:
    state_lower = state.lower()

    # Check for terminal success state
    if re.search(r'success|task completed|accomplished|correct', state_lower):
        return 1.0

    # Check for terminal failure state
    if re.search(r'failed|timeout|error|cannot|impossible', state_lower):
        return 0.0

    # Base probability of success
    score = 0.1

    # Feature 1: Navigation progress (Agent is in a room)
    if re.search(r'in the|go to', state_lower):
        score += 0.15

    # Feature 2: Object Acquisition (Agent is holding the item)
    if re.search(r'holding|pick up', state_lower):
        score += 0.25

    # Feature 3: Object State Change (Item is cleaned)
    if re.search(r'cleaned|clean', state_lower):
        score += 0.20

    # Feature 4: Object Placement (Item is put/located in target)
    if re.search(r'placed|put|on the|in the', state_lower):
        score += 0.25

    # Feature 5: Difficulty (Locked/Broken items increase complexity)
    if re.search(r'locked|broken', state_lower):
        score -= 0.10

    # Clamp the value between 0.0 and 1.0
    score = max(0.0, min(1.0, score))

    return score