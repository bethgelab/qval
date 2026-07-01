import re

def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.0
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Check for goal completion
    success_indicators = ['done', 'completed', 'success', 'finished', 'goal achieved', 'task complete']
    if any(ind in next_state_lower for ind in success_indicators):
        return 1.0
    
    # Check for failure indicators
    failure_indicators = ['cannot', 'nothing', 'empty', 'no', 'fail', 'error', 'already', 'already in']
    if any(ind in next_state_lower for ind in failure_indicators):
        q_value -= 0.2
    
    # Reward productive actions
    productive_actions = ['take', 'put', 'clean', 'heat', 'open', 'close', 'use', 'go to', 'move', 'pick up', 'put in', 'put on']
    if any(act in action_lower for act in productive_actions):
        q_value += 0.15
    
    # Penalize unproductive actions
    unproductive_actions = ['look', 'examine', 'inspect']
    if any(act in action_lower for act in unproductive_actions):
        q_value -= 0.05
    
    # Reward state progression (state changed meaningfully)
    if state_lower != next_state_lower:
        q_value += 0.1
    
    # Check for goal-related objects in state
    goal_objects = ['bottle', 'cup', 'spoon', 'bowl', 'plate', 'sink', 'fridge', 'microwave', 'counter', 'table', 'sofa', 'bed', 'drawer', 'cabinet']
    goal_object_count = sum(1 for obj in goal_objects if obj in state_lower)
    q_value += goal_object_count * 0.02
    
    # Check for location progress
    if 'go to' in action_lower:
        # Navigation is generally productive
        q_value += 0.08
    
    # Check if we're holding an object (progress indicator)
    if 'holding' in next_state_lower or 'in hand' in next_state_lower:
        q_value += 0.1
    
    # Check for object state changes
    state_change_indicators = ['clean', 'dirty', 'open', 'closed', 'hot', 'cold', 'on', 'off']
    state_changes = sum(1 for ind in state_change_indicators if ind in next_state_lower and ind not in state_lower)
    q_value += state_changes * 0.05
    
    # Check for goal keywords in state (we're on track)
    goal_keywords = ['goal', 'target', 'need', 'want', 'should', 'must']
    if any(kw in state_lower for kw in goal_keywords):
        q_value += 0.05
    
    # Penalize being stuck (no progress)
    if state_lower == next_state_lower and 'go to' not in action_lower:
        q_value -= 0.1
    
    # Clamp to reasonable range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value