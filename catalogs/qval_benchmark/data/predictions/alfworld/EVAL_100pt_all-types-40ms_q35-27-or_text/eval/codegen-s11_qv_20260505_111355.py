import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for task completion
    completion_indicators = ['done', 'completed', 'success', 'finished', 'task completed']
    if any(ind in next_state_lower for ind in completion_indicators):
        return 1.0
    
    # Check for failure indicators
    failure_indicators = ['cannot', 'nothing', 'no', 'fail', 'error', 'not found']
    if any(ind in next_state_lower for ind in failure_indicators):
        q_value -= 0.3
    
    # Reward productive actions
    productive_actions = ['take', 'put', 'clean', 'heat', 'open', 'close', 'use', 'go to', 'move']
    if any(act in action_lower for act in productive_actions):
        q_value += 0.1
    
    # Penalize unproductive actions
    unproductive_actions = ['look', 'examine', 'inventory']
    if any(act in action_lower for act in unproductive_actions):
        q_value -= 0.05
    
    # Check if state changed (progress indicator)
    if state_lower != next_state_lower:
        q_value += 0.05
    
    # Check for goal-relevant objects in current state
    goal_objects = ['bottle', 'cup', 'spoon', 'bowl', 'plate', 'sink', 'fridge', 
                   'microwave', 'counter', 'table', 'drawer', 'cabinet']
    object_count = sum(1 for obj in goal_objects if obj in state_lower)
    q_value += object_count * 0.02
    
    # Check if we're in a relevant room
    relevant_rooms = ['kitchen', 'bedroom', 'bathroom', 'living room', 'office']
    if any(room in state_lower for room in relevant_rooms):
        q_value += 0.05
    
    # Clamp to reasonable range
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value