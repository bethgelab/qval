def signal_function(state: str) -> float:
    import re
    
    # Check for terminal success states
    success_indicators = ['success', 'completed', 'task completed', 'task succeeded']
    if any(ind in state.lower() for ind in success_indicators):
        return 1.0
    
    # Check for terminal failure states
    failure_indicators = ['failed', 'timeout', 'exceeded', 'maximum steps']
    if any(ind in state.lower() for ind in failure_indicators):
        return 0.0
    
    # Initialize base value
    value = 0.0
    
    # Extract step count if available (format: "step X of Y" or "step X")
    step_match = re.search(r'step\s*(\d+)', state.lower())
    if step_match:
        steps_used = int(step_match.group(1))
        steps_remaining = 40 - steps_used
        # More steps remaining = more opportunity to succeed
        step_factor = max(0, steps_remaining) / 40.0
        value += step_factor * 0.25
    else:
        # Assume mid-episode if no step info
        value += 0.125
    
    # Check if agent has object in inventory (major progress indicator)
    inventory_indicators = ['inventory', 'holding', 'carrying', 'you have', 'in your inventory']
    has_object = any(ind in state.lower() for ind in inventory_indicators)
    if has_object:
        value += 0.4
    
    # Check if agent is at target location (room keywords)
    location_keywords = ['bedroom', 'kitchen', 'living room', 'bathroom', 'office', 'garage']
    at_target_location = any(loc in state.lower() for loc in location_keywords)
    if at_target_location:
        value += 0.15
    
    # Check for object presence near agent (visible objects)
    visible_indicators = ['you see', 'on the', 'in the', 'near', 'next to']
    has_visible_objects = any(ind in state.lower() for ind in visible_indicators)
    if has_visible_objects:
        value += 0.1
    
    # Check for task completion proximity (object at target container)
    completion_keywords = ['put', 'place', 'on', 'in', 'inside', 'on top of', 'below']
    task_progress = sum(1 for kw in completion_keywords if kw in state.lower())
    value += min(task_progress * 0.03, 0.1)
    
    # Penalize if agent seems stuck (repeated actions or no progress indicators)
    if 'nothing' in state.lower() and 'cannot' in state.lower():
        value -= 0.05
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, value))