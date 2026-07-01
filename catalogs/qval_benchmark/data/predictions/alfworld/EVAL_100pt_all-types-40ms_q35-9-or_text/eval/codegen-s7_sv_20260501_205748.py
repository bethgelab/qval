import re
import math

def signal_function(state: str) -> float:
    # Initialize base value
    value = 0.5
    
    # Check if task appears complete
    if "task done" in state.lower() or "goal achieved" in state.lower() or "completed" in state.lower():
        value = 1.0
        return value
    
    # Check for success indicators
    success_keywords = ["success", "done", "complete", "achieved", "finished", "succeed"]
    for keyword in success_keywords:
        if keyword in state.lower():
            value = 1.0
            return value
    
    # Check for failure indicators
    failure_keywords = ["fail", "error", "impossible", "blocked", "cannot", "not allowed"]
    for keyword in failure_keywords:
        if keyword in state.lower():
            value = 0.1
            return value
    
    # Check for task description and progress
    state_lower = state.lower()
    
    # Check if agent is at correct location
    if "at" in state_lower and ("kitchen" in state_lower or "living" in state_lower or "bedroom" in state_lower or "bathroom" in state_lower or "dining" in state_lower):
        value += 0.1
    
    # Check for object state progress (cleaned, moved, etc.)
    progress_keywords = ["clean", "move", "take", "put", "open", "close", "on", "off"]
    progress_count = 0
    for keyword in progress_keywords:
        # Count occurrences but avoid double counting
        if keyword in state_lower:
            progress_count += 1
    
    if progress_count > 0:
        value += 0.1 * min(progress_count, 3)
    
    # Check for goal object location info
    goal_indicators = ["goal", "target", "destination"]
    goal_found = any(gi in state_lower for gi in goal_indicators)
    if goal_found:
        value += 0.1
    
    # Check for remaining task complexity
    # More task steps mentioned = potentially harder = lower value
    task_step_indicators = ["step", "turn", "move", "action", "need to"]
    complexity = state_lower.count("step") + state_lower.count("move") + state_lower.count("need to")
    value -= 0.05 * min(complexity, 5)
    
    # Check for agent position relative to task
    if "agent" in state_lower and ("at" in state_lower or "in" in state_lower):
        value += 0.05
    
    # Check for object state (clean/dirty)
    if "dirty" in state_lower:
        value -= 0.1
    if "clean" in state_lower:
        value += 0.1
    
    # Check for location proximity hints
    location_keywords = ["kitchen", "living", "bedroom", "bathroom", "dining", "hallway", "corridor"]
    agent_location = state_lower.count("agent") + state_lower.count("i am")
    object_location = state_lower.count("is in") + state_lower.count("located")
    
    if agent_location > 0 and object_location > 0:
        value += 0.05
    
    # Clamp value between 0 and 1
    value = max(0.0, min(1.0, value))
    
    return value