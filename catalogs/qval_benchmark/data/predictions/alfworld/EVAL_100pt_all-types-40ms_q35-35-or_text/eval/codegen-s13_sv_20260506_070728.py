import re

def signal_function(state: str) -> float:
    # Check for explicit success conditions first
    success_patterns = [
        r'\bcompleted\b',
        r'\bsuccess\b',
        r'\bdone\b',
        r'\bsolved\b',
        r'task.*?complete'
    ]
    for pattern in success_patterns:
        if re.search(pattern, state, re.IGNORECASE):
            return 1.0

    # If no explicit success, evaluate progress based on task and state features
    base_score = 0.0
    task_found = False
    goal_location = None
    current_location = None
    holding_object = None
    task_object = None
    
    # Step limit tracking (40 max)
    steps_taken = 0
    step_penalty = 0.0
    
    # 1. Check for Task Definition
    task_match = re.search(r'task.*?:\s*(.*)', state, re.IGNORECASE)
    if task_match:
        task_str = task_match.group(1).strip()
        task_found = True
        
        # Extract goal location (e.g., "to the dining table")
        goal_match = re.search(r'to\s+(?:the\s+)?(\w+(?:\s+\w+)?)', task_str, re.IGNORECASE)
        if goal_match:
            goal_location = goal_match.group(1).lower()
        
        # Extract task object (e.g., "Move the apple")
        obj_match = re.search(r'(?:Move|Put|Clean|Find|Check).*?\s+(?:the\s+)?(\w+)', task_str, re.IGNORECASE)
        if obj_match:
            task_object = obj_match.group(1).lower()

    # 2. Check for Current Location
    loc_match = re.search(r'in\s+(?:the\s+)?(\w+(?:\s+\w+)?)', state, re.IGNORECASE)
    if loc_match:
        current_location = loc_match.group(1).lower()
    
    # 3. Check for Holding Object
    hold_match = re.search(r'\bholding\s+(?:the\s+)?(\w+)', state, re.IGNORECASE)
    if hold_match:
        holding_object = hold_match.group(1).lower()
    
    # 4. Check for Step Count (Penalty for high steps)
    step_match = re.search(r'Step\s+(\d+)', state, re.IGNORECASE)
    if step_match:
        steps_taken = int(step_match.group(1))
        if steps_taken > 30:
            step_penalty = (steps_taken - 30) * 0.02
    
    # 5. Calculate Score
    # Base score if task is known
    if task_found:
        base_score = 0.2
    else:
        base_score = 0.05  # Unknown task progress
        
    # Location Match Bonus
    if current_location and goal_location:
        # Check if current room contains goal room keywords
        if goal_location in current_location or current_location in goal_location:
            base_score += 0.3
        elif current_location == goal_location:
            base_score += 0.3
            
    # Holding Bonus
    if holding_object:
        if task_object and holding_object == task_object:
            base_score += 0.4
        else:
            base_score += 0.1
            
    # Apply Step Penalty
    final_score = base_score - step_penalty
    
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, float(final_score)))