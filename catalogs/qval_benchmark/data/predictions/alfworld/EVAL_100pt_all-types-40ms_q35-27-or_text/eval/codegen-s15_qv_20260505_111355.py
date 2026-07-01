import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for task completion indicators
    completion_indicators = ['done', 'completed', 'success', 'task complete', 'goal achieved', 'finished']
    if any(ind in next_state_lower for ind in completion_indicators):
        return 1.0
    
    # Check for failure/error indicators in next state
    failure_indicators = ['cannot', 'nothing', 'no', 'fail', 'error', 'invalid', 'unable']
    if any(ind in next_state_lower for ind in failure_indicators):
        q_value -= 0.3
    
    # Check for productive actions that typically make progress
    productive_actions = ['take', 'put', 'clean', 'heat', 'open', 'close', 'use', 'go to', 'move', 'pick']
    if any(act in action_lower for act in productive_actions):
        q_value += 0.1
    
    # Check if action resulted in observable state change (progress indicator)
    if state_lower != next_state_lower:
        q_value += 0.05
    
    # Check for goal-related objects in current state
    goal_objects = ['bottle', 'cup', 'spoon', 'bowl', 'plate', 'sink', 'fridge', 'microwave', 'counter', 'table', 'cupboard', 'drawer', 'trashcan']
    object_count = sum(1 for obj in goal_objects if obj in state_lower)
    q_value += object_count * 0.02
    
    # Check for navigation progress
    if 'go to' in action_lower or 'move to' in action_lower:
        q_value += 0.05
    
    # Check if we're in a relevant room for household tasks
    relevant_rooms = ['kitchen', 'bedroom', 'bathroom', 'living room', 'office', 'dining room']
    if any(room in state_lower for room in relevant_rooms):
        q_value += 0.03
    
    # Bonus for actions that manipulate objects (more likely to complete task)
    if any(manip in action_lower for manip in ['take', 'put', 'move', 'clean', 'heat']):
        q_value += 0.08
    
    # Penalty for looking/observing actions (less productive)
    if any(obs in action_lower for obs in ['look', 'examine', 'observe', 'inspect']):
        q_value -= 0.05
    
    # Check for positive state transitions (e.g., object moved to desired location)
    positive_transitions = ['now', 'successfully', 'placed', 'moved', 'put in', 'put on']
    if any(trans in next_state_lower for trans in positive_transitions):
        q_value += 0.1
    
    # Clamp to reasonable range
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value