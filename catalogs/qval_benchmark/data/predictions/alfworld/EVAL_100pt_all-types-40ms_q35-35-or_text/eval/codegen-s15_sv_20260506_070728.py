import re

def signal_function(state: str) -> float:
    # Check for explicit success indicators
    if re.search(r'\bsuccess\b|\bcomplete\b|\btask completed\b|\b1\.0\b', state, re.I):
        return 1.0
    
    # Check for explicit failure indicators
    if re.search(r'\bfailed\b|\btimeout\b|\b0\.0\b|\binvalid\b|\berror\b', state, re.I):
        return 0.0
    
    # Extract step count to apply time penalty
    step_match = re.search(r'step[:\s]*(\d+)', state, re.I)
    current_step = 0
    if step_match:
        try:
            current_step = int(step_match.group(1))
        except ValueError:
            current_step = 0
    
    # Calculate step factor: value decreases as steps approach limit (40)
    # If steps >= 40, factor is 0. Otherwise, linear decay.
    if current_step >= 40:
        step_factor = 0.0
    else:
        step_factor = max(0.0, 1.0 - (current_step / 40.0))
    
    # Extract task description
    task_match = re.search(r'task[:\s]*(.+?)(?=\n|$)', state, re.I | re.S)
    if not task_match:
        # No task info, return neutral value adjusted by steps
        return max(0.0, min(1.0, 0.5 * step_factor))
    
    task_desc = task_match.group(1).strip()
    progress = 0.0
    
    # Analyze task type and current state alignment
    # Priority 1: "put" tasks (move object to location)
    if re.search(r'put\s+', task_desc, re.I):
        # Extract target object and location from task
        goal_match = re.search(r'put\s+(\w+)\s+(?:in|on)\s+the\s+(\w+)', task_desc, re.I)
        target_obj = None
        target_loc = None
        if goal_match:
            target_obj = goal_match.group(1).lower()
            target_loc = goal_match.group(2).lower()
        
        # Check if object is already in target location
        obj_loc_match = re.search(r'(\w+)\s+(?:on|in)\s+the\s+(\w+)', state, re.I)
        obj_on_target = False
        if obj_loc_match and target_obj and target_loc:
            curr_obj = obj_loc_match.group(1).lower()
            curr_loc = obj_loc_match.group(2).lower()
            if curr_obj == target_obj and curr_loc == target_loc:
                obj_on_target = True
        
        # Check if agent is holding the target object
        hold_match = re.search(r'holding\s+(\w+)', state, re.I)
        holding_obj = False
        if hold_match and target_obj:
            holding_obj = hold_match.group(1).lower() == target_obj
        
        # Check if agent is in the target location
        agent_loc_match = re.search(r'in\s+the\s+(\w+)', state, re.I)
        in_target_loc = False
        if agent_loc_match and target_loc:
            in_target_loc = agent_loc_match.group(1).lower() == target_loc
        
        # Compute progress based on alignment
        if obj_on_target:
            progress = 0.95  # Task almost done
        elif holding_obj and in_target_loc:
            progress = 0.85  # Ready to execute final action
        elif holding_obj:
            progress = 0.5   # Have object, need to move
        elif in_target_loc:
            progress = 0.3   # In location, need object
        else:
            progress = 0.1   # Far from goal
            
    # Priority 2: "clean" tasks
    elif re.search(r'clean\s+', task_desc, re.I):
        # Check if object is described as clean
        clean_match = re.search(r'clean\s+(\w+)', state, re.I)
        if clean_match:
            progress = 0.8
        else:
            progress = 0.2
            
    # Priority 3: "heat" or "cool" tasks (simplified)
    elif re.search(r'(heat|cool)\s+', task_desc, re.I):
        # Check if object is in stove/microfridge context or heated state
        # Heuristic: if state mentions "stove" or "microwave" and "on", assume progress
        if re.search(r'(stove|microwave|fridge)', state, re.I):
            progress = 0.4
        else:
            progress = 0.1
            
    # Default for other tasks
    else:
        progress = 0.5
        
    # Combine progress with step factor
    final_value = progress * step_factor
    
    # Clamp result to [0.0, 1.0]
    return max(0.0, min(1.0, float(final_value)))