import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for ALFWorld environment based on state, action, and next_state.
    
    Q-value represents expected discounted cumulative reward from this state-action pair.
    Uses heuristic analysis of text patterns to estimate task progress without simulation.
    """
    
    # Base Q-value
    q_value = 0.0
    
    # 1. Check if task is completed (terminal state with success)
    completion_indicators = ['success', 'task', 'goal', 'completed', 'done', 'correct']
    for indicator in completion_indicators:
        if indicator in next_state.lower():
            q_value += 0.5
            break
    
    # 2. Check if action is a goal-completing action
    goal_actions = ['put', 'clean', 'cook', 'eat', 'use', 'wash']
    action_lower = action.lower()
    
    for ga in goal_actions:
        if ga in action_lower:
            q_value += 0.15
            break
    
    # 3. Check for location-based progress (moving objects to target locations)
    location_words = ['in', 'on', 'at', 'to', 'from', 'into']
    location_count = sum(1 for lw in location_words if lw in action_lower)
    q_value += location_count * 0.05
    
    # 4. Check if action matches common ALFWorld movement patterns
    movement_patterns = [
        r'put\s+\S+\s+(?:in|on|at)\s+\S+',
        r'take\s+\S+\s+(?:from|off|on)\s+\S+',
        r'clean\s+\S+',
        r'cook\s+\S+',
    ]
    
    for pattern in movement_patterns:
        if re.search(pattern, action_lower):
            q_value += 0.1
            break
    
    # 5. Check if next_state shows object movement (progress indicator)
    task_objects = ['plate', 'cup', 'bowl', 'somethings', 'thing', 'object', 'book', 'knife', 'fork', 'spoon', 'pencil', 'pen']
    task_locations = ['diningtable', 'desk', 'countertop', 'sink', 'microwave', 'oven', 'fridge', 'cabinet', 'drawer', 'shelf']
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Check if new objects appear in next state
    for obj in task_objects:
        if obj in next_state_lower and obj not in state_lower:
            q_value += 0.05
            break
    
    # Check if task-relevant locations appear in next state
    for loc in task_locations:
        if loc in next_state_lower and loc not in state_lower:
            q_value += 0.05
            break
    
    # 6. Bonus for valid navigation actions
    nav_actions = ['go to', 'go', 'walk', 'walk to', 'move', 'examine']
    for nav in nav_actions:
        if nav in action_lower:
            q_value += 0.05
            break
    
    # 7. Reward for state changes that indicate progress
    # Look for action result indicators in next_state
    result_indicators = ['you', 'your', 'placed', 'moved', 'taken', 'cleaned', 'cooked']
    for ri in result_indicators:
        if ri in next_state_lower:
            q_value += 0.05
            break
    
    # 8. Ensure Q-value is in valid range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value