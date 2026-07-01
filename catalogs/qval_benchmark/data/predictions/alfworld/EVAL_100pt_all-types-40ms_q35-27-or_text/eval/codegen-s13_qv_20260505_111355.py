import re

def signal_function(state: str, action: str, next_state: str) -> float:
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for task completion indicators
    completion_keywords = ['done', 'completed', 'success', 'finished', 'goal achieved', 'task complete']
    if any(kw in next_state_lower for kw in completion_keywords):
        return 1.0
    
    # Check for failure indicators
    failure_keywords = ['cannot', 'nothing', 'no way', 'fail', 'error', 'impossible', "can't"]
    if any(kw in next_state_lower for kw in failure_keywords):
        return -0.5
    
    # Base score
    q_value = 0.0
    
    # Reward productive actions
    productive_actions = ['take', 'put', 'clean', 'heat', 'open', 'close', 'use', 'go to', 'move', 'pick', 'put in', 'put on', 'turn on', 'turn off']
    if any(act in action_lower for act in productive_actions):
        q_value += 0.15
    
    # Penalize redundant actions (state didn't change)
    if state_lower == next_state_lower:
        q_value -= 0.3
    
    # Reward navigation actions
    if 'go to' in action_lower or 'move to' in action_lower:
        q_value += 0.1
    
    # Check for progress indicators in next state
    progress_keywords = ['now', 'has', 'is', 'on', 'in', 'with', 'contains', 'found']
    progress_count = sum(1 for kw in progress_keywords if kw in next_state_lower)
    q_value += progress_count * 0.02
    
    # Check for goal-related objects in state
    goal_objects = ['bottle', 'cup', 'spoon', 'bowl', 'plate', 'sink', 'fridge', 'microwave', 'counter', 'table', 'stove', 'oven', 'trashcan', 'toaster', 'coffee', 'water']
    object_count = sum(1 for obj in goal_objects if obj in state_lower)
    q_value += object_count * 0.03
    
    # Penalize negative feedback in next state
    negative_feedback = ['nothing', 'empty', 'no', 'not', "doesn't", "didn't", "can't", 'cannot']
    negative_count = sum(1 for neg in negative_feedback if neg in next_state_lower)
    q_value -= negative_count * 0.1
    
    # Reward if action appears to have changed something
    if state_lower != next_state_lower:
        # Check for object state changes
        if any(word in next_state_lower and word not in state_lower for word in ['taken', 'clean', 'heated', 'opened', 'closed', 'filled', 'poured']):
            q_value += 0.2
    
    # Bonus for actions that seem to advance toward typical goals
    if any(kw in action_lower for kw in ['clean', 'heat', 'put', 'take']):
        q_value += 0.1
    
    # Cap the Q-value to reasonable range
    q_value = max(-1.0, min(1.0, q_value))
    
    return q_value