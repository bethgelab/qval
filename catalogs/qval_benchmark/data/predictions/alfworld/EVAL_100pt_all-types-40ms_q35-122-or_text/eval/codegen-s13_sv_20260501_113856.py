def signal_function(state: str) -> float:
    import re
    
    state_lower = state.lower()
    
    # Terminal success - maximum value
    success_terms = ['success', 'completed', 'done', 'finished', 'task complete']
    if any(term in state_lower for term in success_terms):
        return 1.0
    
    # Terminal failure - zero value
    failure_terms = ['failed', 'timeout', 'exceeded', 'limit reached', 'cannot']
    if any(term in state_lower for term in failure_terms):
        return 0.0
    
    # Base value for being in a valid state
    value = 0.1
    
    # Check if agent is holding an object (progress indicator)
    holding_terms = ['holding', 'have', 'in your hand', 'inventory', 'carrying']
    if any(term in state_lower for term in holding_terms):
        value += 0.25
    
    # Check for object placement actions (major progress toward goal)
    placement_terms = ['put', 'place', 'on the', 'in the', 'at the', 'located on', 'located in']
    if any(term in state_lower for term in placement_terms):
        value += 0.25
    
    # Check for navigation between rooms (progress indicator)
    room_terms = ['kitchen', 'bedroom', 'bathroom', 'living room', 'dining room', 'office', 'garage', 'hallway', 'basement']
    room_count = sum(1 for room in room_terms if room in state_lower)
    value += min(room_count * 0.05, 0.15)
    
    # Check for object interaction verbs (progress indicator)
    interaction_terms = ['open', 'close', 'turn on', 'turn off', 'clean', 'wash', 'dry', 'heat', 'cook', 'cut', 'slice']
    if any(term in state_lower for term in interaction_terms):
        value += 0.1
    
    # Check for task-relevant object mentions (indicates awareness of goal)
    common_objects = ['plate', 'cup', 'bowl', 'book', 'pen', 'key', 'apple', 'banana', 'tomato', 'potato', 
                      'egg', 'bread', 'milk', 'water', 'dish', 'cloth', 'towel', 'lamp', 'light', 
                      'switch', 'door', 'drawer', 'cabinet', 'fridge', 'microwave', 'sink', 'toilet',
                      'bed', 'sofa', 'chair', 'table', 'desk', 'shelf', 'box', 'bag', 'suitcase']
    object_count = sum(1 for obj in common_objects if obj in state_lower)
    value += min(object_count * 0.03, 0.15)
    
    # Check for step count to estimate remaining time
    step_match = re.search(r'step\s*(\d+)', state_lower)
    if step_match:
        current_step = int(step_match.group(1))
        remaining_ratio = max(0, (40 - current_step) / 40.0)
        value += remaining_ratio * 0.1
    
    # Cap value at 1.0
    return min(value, 1.0)