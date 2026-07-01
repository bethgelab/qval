import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Check for goal completion - highest value
    completion_indicators = ['done', 'completed', 'success', 'finished', 'goal achieved', 'task complete']
    if any(ind in next_state_lower for ind in completion_indicators):
        return 1.0
    
    # Check for failure/error indicators - penalize
    failure_indicators = ['cannot', 'nothing', 'error', 'fail', 'not found', 'does not exist']
    if any(ind in next_state_lower for ind in failure_indicators):
        q_value -= 0.3
    
    # Reward productive actions that change state
    productive_actions = ['take', 'put', 'clean', 'heat', 'open', 'close', 'use', 'go to', 'move to', 'pick up', 'drop']
    if any(act in action_lower for act in productive_actions):
        q_value += 0.15
    
    # Penalize unproductive/redundant actions
    unproductive_actions = ['look', 'examine', 'inventory']
    if any(act in action_lower for act in unproductive_actions):
        q_value -= 0.05
    
    # Reward state change (progress indicator)
    if state_lower != next_state_lower:
        q_value += 0.1
    
    # Penalize no state change (wasted action)
    if state_lower == next_state_lower:
        q_value -= 0.2
    
    # Check for goal-related objects in current state
    goal_objects = ['bottle', 'cup', 'spoon', 'bowl', 'plate', 'sink', 'fridge', 'microwave', 
                    'counter', 'table', 'drawer', 'cabinet', 'stove', 'oven', 'toaster']
    object_count = sum(1 for obj in goal_objects if obj in state_lower)
    q_value += object_count * 0.02
    
    # Check if in relevant room
    relevant_rooms = ['kitchen', 'bedroom', 'bathroom', 'living room', 'office']
    if any(room in state_lower for room in relevant_rooms):
        q_value += 0.05
    
    # Check for positive state changes (object acquired, location changed)
    positive_changes = ['you have', 'you picked', 'you took', 'you are now', 'in the']
    if any(chg in next_state_lower for chg in positive_changes):
        q_value += 0.1
    
    # Check for navigation progress
    if 'go to' in action_lower or 'move to' in action_lower:
        # Check if location actually changed
        if 'you are' in state_lower and 'you are' in next_state_lower:
            # Extract locations and check if different
            loc_pattern = r'you are in the (\w+)'
            state_loc = re.search(loc_pattern, state_lower)
            next_loc = re.search(loc_pattern, next_state_lower)
            if state_loc and next_loc and state_loc.group(1) != next_loc.group(1):
                q_value += 0.1
    
    # Clamp to reasonable range
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value