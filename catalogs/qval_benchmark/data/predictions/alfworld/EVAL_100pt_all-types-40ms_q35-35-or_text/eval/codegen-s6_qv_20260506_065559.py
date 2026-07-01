import re

def signal_function(state: str, action: str, next_state: str) -> float:
    # Check for explicit success indicators in the next state
    success_keywords = ["task is complete", "success", "goal reached", "congratulations", "completed"]
    for keyword in success_keywords:
        if keyword in next_state.lower():
            return 1.0

    # If no state change occurred, progress is zero
    if state == next_state:
        return 0.0

    # Check if action is valid/meaningful
    action_lower = action.lower().strip()
    if not action_lower:
        return 0.0

    # Heuristic: Count object relations (progress indicators)
    # These patterns indicate objects are being placed or manipulated
    relation_patterns = ["is on the", "is in the", "is cleaned", "is wet", "is full", "is broken"]
    
    state_relation_count = 0
    next_state_relation_count = 0
    
    for pattern in relation_patterns:
        state_relation_count += len(re.findall(pattern, state.lower()))
        next_state_relation_count += len(re.findall(pattern, next_state.lower()))
    
    progress_delta = next_state_relation_count - state_relation_count
    
    # Heuristic: Action type relevance
    # 'put', 'place', 'clean' are closer to goal than 'go', 'take'
    action_score = 0.0
    if any(cmd in action_lower for cmd in ["put", "place", "clean"]):
        action_score += 0.3
    elif any(cmd in action_lower for cmd in ["take", "go", "open", "close"]):
        action_score += 0.1
    elif any(cmd in action_lower for cmd in ["fill", "empty", "wash"]):
        action_score += 0.2

    # Calculate Q-value estimate
    # Base value is 0.0. Success is 1.0.
    # Progress adds value. Action relevance adds value.
    q_value = 0.0
    
    # Reward significant progress in state structure
    if progress_delta > 0:
        q_value += 0.25 * progress_delta
    
    # Reward relevant actions
    q_value += action_score
    
    # If we are making progress but not yet at success, Q is between 0 and 1
    # Clamp the value to valid probability range
    return min(1.0, max(0.0, q_value))