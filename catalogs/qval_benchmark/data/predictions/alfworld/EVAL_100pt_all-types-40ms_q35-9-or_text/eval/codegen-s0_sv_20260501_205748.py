import re
import math

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Terminal success indicators
    success_terms = ["completed", "done", "success", "goal achieved", "task complete", "successfully"]
    if any(term in state_lower for term in success_terms):
        return 0.95
    
    # Terminal failure indicators
    failure_terms = ["failed", "error", "cannot", "impossible", "blocked", "unreachable"]
    if any(term in state_lower for term in failure_terms):
        return 0.05
    
    # Check for step information
    step_match = re.search(r"step\s*(\d+)", state_lower)
    if step_match:
        try:
            step_num = int(step_match.group(1))
            if step_num > 0:
                value = 1.0 - (step_num / 40.0)
            else:
                value = 0.5
        except (ValueError, AttributeError):
            value = 0.5
    else:
        value = 0.5
    
    # Progress indicators
    progress_terms = ["closer", "near", "almost", "ready", "can take"]
    if any(term in state_lower for term in progress_terms):
        value += 0.15
    
    # Negative progress indicators
    negative_terms = ["far", "away", "blocked", "unreachable", "not accessible"]
    if any(term in state_lower for term in negative_terms):
        value -= 0.15
    
    # Goal mention
    goal_keywords = ["goal", "target", "task", "objective", "need to"]
    if any(kw in state_lower for kw in goal_keywords):
        value += 0.05
    
    # Object accessibility
    if "accessible" in state_lower or "reachable" in state_lower:
        value += 0.1
    if "unreachable" in state_lower or "not accessible" in state_lower:
        value -= 0.1
    
    # Location proximity
    if "at the" in state_lower or "in the" in state_lower:
        value += 0.05
    
    # Clamp value between 0 and 1
    value = max(0.0, min(1.0, value))
    
    return round(value, 4)