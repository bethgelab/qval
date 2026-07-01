def signal_function(state: str) -> float:
    """
    Estimate the state-value V(s) for an ALFWorld environment state.
    
    Based on analysis of state text for task completion, progress indicators,
    and potential blockers. Returns a float in [0, 1] representing expected
    discounted cumulative reward.
    """
    state_lower = state.lower()
    
    # Check if task is already completed
    completion_keywords = [
        "successfully", "completed", "task completed", "done", "finished",
        "goal achieved", "all done", "success"
    ]
    if any(kw in state_lower for kw in completion_keywords):
        return 1.0
    
    # Check for failure indicators
    failure_keywords = [
        "failed", "error", "cannot", "impossible", "not allowed",
        "blocked", "out of reach", "broken", "missing"
    ]
    if any(kw in state_lower for kw in failure_keywords):
        return 0.0
    
    # Check for progress indicators (objects being handled)
    progress_keywords = [
        "holding", "picking up", "placing", "putting", "moving",
        "carrying", "opening", "closing", "cleaning", "washing"
    ]
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    
    # Check for location information
    location_keywords = [
        "kitchen", "living room", "bedroom", "bathroom", "hallway",
        "table", "floor", "shelf", "cabinet", "counter", "sink"
    ]
    location_count = sum(1 for kw in location_keywords if kw in state_lower)
    
    # Check for task-related objects
    object_keywords = [
        "mug", "bottle", "glass", "cup", "plate", "book", "box",
        "toy", "phone", "lamp", "plant", "remote", "key", "wallet"
    ]
    object_count = sum(1 for kw in object_keywords if kw in state_lower)
    
    # Base value starts at 0.5 (neutral state)
    base_value = 0.5
    
    # Adjust based on progress indicators
    if progress_count >= 2:
        base_value += 0.15
    elif progress_count == 1:
        base_value += 0.05
    
    # Adjust based on location clarity (more locations = better navigation)
    if location_count >= 3:
        base_value += 0.05
    elif location_count >= 2:
        base_value += 0.02
    
    # Adjust based on object mentions (more objects = task is active)
    if object_count >= 3:
        base_value += 0.05
    elif object_count >= 2:
        base_value += 0.02
    
    # Cap value between 0 and 1
    base_value = max(0.0, min(1.0, base_value))
    
    return base_value