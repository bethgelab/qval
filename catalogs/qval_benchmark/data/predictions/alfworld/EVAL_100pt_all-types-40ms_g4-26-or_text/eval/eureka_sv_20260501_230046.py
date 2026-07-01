import re

def signal_function(state: str):
    """
    Estimates the state-value for an ALFWorld agent by parsing the goal from the state 
    and determining progress across different task types (move, clean, find).
    Ensures a consistent, dense, and monotonic signal with a stable set of components.
    """
    state_lower = state.lower()
    obs_part = state_lower
    goal_part = ""
    
    # 1. Split Observation and Goal
    if "goal:" in state_lower:
        obs_part, goal_part = state_lower.split("goal:", 1)
    elif "task:" in state_lower:
        obs_part, goal_part = state_lower.split("task:", 1)
    
    obs_part = obs_part.strip()
    goal_part = goal_part.strip()

    # Initialize standardized components
    success = 0.0
    obj_progress = 0.0
    loc_progress = 0.0
    base_value = 0.05

    # 2. Success Check (General)
    success_keywords = ["successfully", "completed", "task finished", "achieved"]
    if any(k in obs_part for k in success_keywords):
        success = 1.0

    # 3. Goal Parsing
    task_type = None
    target_obj = None
    target_loc = None

    if goal_part:
        # Move/Put/Place/Drop: "put the apple in the fridge"
        m_move = re.search(r"(?:put|move|place|drop)\s+(?:the|an|a)?\s*(.+?)\s+(?:in|inside|on|at|to|into)\s+(?:the|an|a)?\s*(.+)", goal_part)
        # Clean/Wash/Scrub: "clean the plate"
        m_clean = re.search(r"(?:clean|wash|scrub)\s+(?:the|an|a)?\s*(.+)", goal_part)
        # Find/Get/Locate: "find the key"
        m_find = re.search(r"(?:find|get|locate)\s+(?:the|an|a)?\s*(.+)", goal_part)

        if m_move:
            task_type = "move"
            target_obj = re.sub(r"^(?:the|an|a)\s+", "", m_move.group(1).strip())
            target_loc = re.sub(r"^(?:the|an|a)\s+", "", m_move.group(2).strip())
        elif m_clean:
            task_type = "clean"
            target_obj = re.sub(r"^(?:the|an|a)\s+", "", m_clean.group(1).strip())
        elif m_find:
            task_type = "find"
            target_obj = re.sub(r"^(?:the|an|a)\s+", "", m_find.group(1).strip())

    # 4. Progress Estimation
    if target_obj:
        obj_esc = re.escape(target_obj)
        # Check if holding or if object is visible (handling plurals with s?)
        is_holding = bool(re.search(rf"holding\s+(?:the|an|a)?\s*{obj_esc}s?\b", obs_part))
        is_visible = bool(re.search(rf"\b{obj_esc}s?\b", obs_part))
        
        if task_type == "move" and target_loc:
            loc_esc = re.escape(target_loc)
            # Check for object at target location
            if re.search(rf"{obj_esc}s?.*?\s+(?:in|inside|on|at)\s+(?:the|an|a)?\s*{loc_esc}", obs_part, re.DOTALL):
                success = 1.0
            
            # Location proximity
            if is_holding and re.search(rf"\b{loc_esc}s?\b", obs_part):
                loc_progress = 0.4
            elif re.search(rf"\b{loc_esc}s?\b", obs_part):
                loc_progress = 0.2

        elif task_type == "clean":
            # Check if object is described as clean
            if re.search(rf"{obj_esc}s?.*?\s+(?:clean|wash|scrub|polished)", obs_part, re.DOTALL):
                success = 1.0
        
        elif task_type == "find":
            # For find tasks, holding the item is often the success condition
            if is_holding:
                success = 1.0

        # Object progress if not yet successful
        if success < 1.0:
            if is_holding:
                obj_progress = 0.4
            elif is_visible:
                obj_progress = 0.2
                
    elif goal_part:
        # Goal exists but parsing failed: provide a generic object-presence signal
        obj_count = len(re.findall(r"(?:a|an|the)\s+\w+", obs_part))
        obj_progress = min(0.3, 0.05 * obj_count)
    else:
        # No goal parsed at all: fallback to environment density
        obj_count = len(re.findall(r"(?:a|an|the)\s+\w+", obs_part))
        rel_count = len(re.findall(r"\b(?:in|inside|on|at)\b", obs_part))
        obj_progress = min(0.3, 0.05 * obj_count)
        loc_progress = min(0.1, 0.05 * rel_count)

    # 5. Final Score calculation
    # Maintain stability by ensuring a consistent set of keys in the return dictionary.
    if success >= 1.0:
        total = 1.0
        components = {
            "success": 1.0,
            "object_progress": 0.0,
            "location_progress": 0.0,
            "base_value": 0.0
        }
    else:
        total = success + obj_progress + loc_progress + base_value
        if total > 1.0:
            scale = 1.0 / total
            components = {
                "success": success * scale,
                "object_progress": obj_progress * scale,
                "location_progress": loc_progress * scale,
                "base_value": base_value * scale
            }
            total = 1.0
        else:
            components = {
                "success": success,
                "object_progress": obj_progress,
                "location_progress": loc_progress,
                "base_value": base_value
            }

    return total, components