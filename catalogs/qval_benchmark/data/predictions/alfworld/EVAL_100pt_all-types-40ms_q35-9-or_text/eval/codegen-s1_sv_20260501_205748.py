import re
from collections import Counter

def signal_function(state: str) -> float:
    """
    Estimates the state-value for a given state in ALFWorld.
    Returns a float between 0.0 and 1.0 representing the estimated
    probability-weighted value of reaching the goal from this state.
    """
    state_lower = state.lower()
    
    # Base value for neutral state
    value = 0.5
    
    # Check for explicit completion signals
    completion_patterns = [
        r'\bcompleted\b', r'\bdone\b', r'\bfinished\b', 
        r'\bsuccess\b', r'\bgoal achieved\b', r'\btask complete\b',
        r'\bsuccessful\b', r'\ball done\b'
    ]
    if any(re.search(p, state_lower) for p in completion_patterns):
        return 0.98
    
    # Check for explicit failure or blocking signals
    failure_patterns = [
        r'\bfailed\b', r'\berror\b', r'\bcannot\b', r'\bimpossible\b',
        r'\bblocked\b', r'\bunreachable\b', r'\btoo many\b',
        r'\bstep limit\b', r'\btimeout\b'
    ]
    if any(re.search(p, state_lower) for p in failure_patterns):
        return 0.15
    
    # Extract task goal information
    goal_keywords = [r'\bgoal\b', r'\btake\b', r'\bmove\b', r'\bbring\b', 
                     r'\bopen\b', r'\bclean\b', r'\bwash\b', r'\bcook\b',
                     r'\bfold\b', r'\bsort\b', r'\bplace\b', r'\bput\b']
    goal_count = sum(1 for p in goal_keywords if re.search(p, state_lower))
    
    # Count objects mentioned (more objects = potentially more complex task)
    object_patterns = [
        r'\bcup\b', r'\bbottle\b', r'\bplate\b', r'\bspoon\b', r'\bfork\b',
        r'\bknife\b', r'\bmug\b', r'\bbook\b', r'\blamp\b', r'\bpillow\b',
        r'\bremote\b', r'\bphone\b', r'\btoy\b', r'\bclothes\b', r'\bbed\b',
        r'\bsofa\b', r'\bchair\b', r'\btable\b', r'\bdoor\b', r'\bwindow\b',
        r'\bbag\b', r'\btrash\b', r'\bcleaning\b', r'\bwashing\b'
    ]
    object_count = sum(1 for p in object_patterns if re.search(p, state_lower))
    
    # Count location mentions (more locations = potentially more navigation)
    location_patterns = [
        r'\bkitchen\b', r'\bliving room\b', r'\bbedroom\b', r'\bbathroom\b',
        r'\bhallway\b', r'\bdoor\b', r'\btable\b', r'\bchair\b', r'\bbed\b',
        r'\bsofa\b', r'\bwashing machine\b', r'\bdishwasher\b', r'\bfridge\b'
    ]
    location_count = sum(1 for p in location_patterns if re.search(p, state_lower))
    
    # Check for progress indicators (holding, carrying, opened, cleaned, etc.)
    progress_patterns = [
        r'\bholding\b', r'\bcarrying\b', r'\bpicked up\b', r'\bopened\b',
        r'\bcleaned\b', r'\bwashed\b', r'\bfolded\b', r'\bplaced\b',
        r'\bput on\b', r'\bput in\b', r'\bput under\b', r'\bput near\b'
    ]
    progress_count = sum(1 for p in progress_patterns if re.search(p, state_lower))
    
    # Check for "at" or "in" prepositions indicating current position
    position_patterns = [r'\bat\b', r'\bin\b', r'\bon\b', r'\bunder\b']
    position_count = sum(1 for p in position_patterns if re.search(p, state_lower))
    
    # Check for step/turn information
    step_patterns = [r'\bstep\b', r'\bturn\b', r'\bmove\b']
    step_mentions = sum(1 for p in step_patterns if re.search(p, state_lower))
    
    # Heuristic scoring based on features
    
    # More progress = higher value
    progress_bonus = min(progress_count * 0.05, 0.3)
    value += progress_bonus
    
    # More objects to handle = potentially more complex task
    complexity_penalty = min(object_count * 0.02, 0.15)
    value -= complexity_penalty
    
    # More locations = more navigation needed
    navigation_penalty = min(location_count * 0.015, 0.1)
    value -= navigation_penalty
    
    # Position mentions indicate clear current state
    position_bonus = min(position_count * 0.03, 0.1)
    value += position_bonus
    
    # Step mentions might indicate awareness of progress
    step_bonus = min(step_mentions * 0.02, 0.05)
    value += step_bonus
    
    # Goal keyword presence suggests clear objective
    goal_bonus = min(goal_count * 0.04, 0.1)
    value += goal_bonus
    
    # Cap value between 0 and 1
    value = max(0.0, min(1.0, value))
    
    return value