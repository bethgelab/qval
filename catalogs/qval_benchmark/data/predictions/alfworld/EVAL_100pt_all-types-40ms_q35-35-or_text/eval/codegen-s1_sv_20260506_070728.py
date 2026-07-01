import re

def signal_function(state: str) -> float:
    s = state.lower()
    
    # 1. Check for immediate success
    if any(term in s for term in ["success", "completed", "done", "task complete"]):
        return 1.0
    
    # 2. Check step count if available (decay value near limit)
    step_match = re.search(r'step\s+(\d+)', s)
    if step_match:
        current_step = int(step_match.group(1))
        if current_step >= 38:
            return 0.0  # Critical failure risk
    
    # 3. Extract Task Goal
    task_match = re.search(r'task:\s*(.+?)(?:\n|$)', s, re.IGNORECASE)
    if not task_match:
        # Fallback: check for generic progress indicators
        if any(term in s for term in ["cleaned", "cooked", "charged", "washed"]):
            return 0.8
        return 0.2
    
    task_text = task_match.group(1)
    
    # 4. Analyze specific task types
    # Type: Move [obj] to [loc]
    move_match = re.search(r'move the (.+?) to the (.+?)', task_text)
    if move_match:
        obj = move_match.group(1)
        loc = move_match.group(2)
        
        score = 0.2  # Base value
        
        # Check if holding object
        # Heuristic: "holding" appears and object name is near it
        holding_idx = s.find("holding")
        obj_idx = s.find(obj)
        if holding_idx != -1 and obj_idx != -1 and abs(holding_idx - obj_idx) < 50:
            score += 0.3
        
        # Check if agent is at location
        # Heuristic: "in the [loc]"
        if re.search(r'in the ' + re.escape(loc) + r'\b', s):
            score += 0.3
            
        # Check if object is already at location
        # Heuristic: "on the [loc]" or "at the [loc]"
        if re.search(r'(on|at) the ' + re.escape(loc) + r'\b', s):
            score += 0.5
            
        return min(max(score, 0.0), 1.0)
    
    # Type: Clean/Cook/Charge [obj]
    # Check if the object state is correct
    if "clean" in task_text:
        if "cleaned" in s or "clean" in s and "dirty" not in s:
            return 1.0
        if "sink" in s or "clean" in s:
            return 0.6
        return 0.3
        
    if "cook" in task_text:
        if "cooked" in s:
            return 1.0
        if "stove" in s or "microwave" in s:
            return 0.6
        return 0.3
        
    if "charge" in task_text:
        if "charged" in s:
            return 1.0
        if "charger" in s or "outlet" in s:
            return 0.6
        return 0.3

    # Default fallback
    return 0.2