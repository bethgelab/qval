import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value based on state features, action, and next state.
    
    Returns a float in [0, 1] representing expected return from this state-action pair.
    """
    # Base score starts neutral
    score = 0.5
    
    # Check if action is a goal-achieving action (terminal state indicator)
    goal_indicators = [
        'completed', 'success', 'done', 'sent', 'saved', 'added',
        'created', 'updated', 'finished', 'goal_achieved', 'task_complete'
    ]
    
    # Check if next state shows goal achievement
    next_state_lower = next_state.lower()
    goal_achieved = any(ind in next_state_lower for ind in goal_indicators)
    
    if goal_achieved:
        return 1.0
    
    # Analyze state features to understand current task context
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Task type indicators
    task_types = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor']
    current_task = None
    for task in task_types:
        if task in state_lower:
            current_task = task
            break
    
    # Action quality scoring
    # Good actions: clicking relevant elements, filling forms, pressing submit
    good_actions = ['click', 'fill', 'press', 'submit', 'add', 'create']
    action_type = 'unknown'
    for ga in good_actions:
        if ga in action_lower:
            action_type = ga
            break
    
    # Reward appropriate actions for current task
    if current_task == 'todo':
        if 'add' in action_lower or 'create' in action_lower:
            score += 0.15
        if 'click' in action_lower and 'button' in action_lower:
            score += 0.1
    elif current_task == 'calendar':
        if 'add' in action_lower or 'event' in action_lower:
            score += 0.15
        if 'click' in action_lower and ('event' in action_lower or 'day' in action_lower):
            score += 0.1
    elif current_task == 'messenger':
        if 'send' in action_lower or 'message' in action_lower:
            score += 0.15
        if 'fill' in action_lower and 'message' in action_lower:
            score += 0.1
    elif current_task == 'maps':
        if 'search' in action_lower or 'navigate' in action_lower:
            score += 0.15
        if 'click' in action_lower and ('location' in action_lower or 'point' in action_lower):
            score += 0.1
    elif current_task == 'code' or current_task == 'editor':
        if 'edit' in action_lower or 'write' in action_lower:
            score += 0.15
        if 'click' in action_lower and 'editor' in action_lower:
            score += 0.1
    
    # Progress check: compare state to next_state
    # If next_state has more form fields filled or more elements visible, reward
    state_elements = re.findall(r'\[bid:\d+\]', state_lower)
    next_state_elements = re.findall(r'\[bid:\d+\]', next_state_lower)
    
    # More interactive elements in next state might indicate progress
    if len(next_state_elements) > len(state_elements):
        score += 0.05
    
    # Check for form completion indicators in next state
    form_indicators = ['filled', 'submitted', 'entered', 'typed', 'completed']
    if any(ind in next_state_lower for ind in form_indicators):
        score += 0.1
    
    # Penalty for noop actions (unless at end of task)
    if 'noop' in action_lower:
        score -= 0.05
    
    # Penalty for clicking wrong elements (if action doesn't match visible elements)
    if 'click' in action_lower:
        bid_match = re.search(r'click\([\'"]?bid([\'"]?\s*:\s*\d+[\'"]?)', action_lower)
        if bid_match:
            bid_num = bid_match.group(1)
            # Check if this bid appears in next state (element still exists)
            if bid_num not in next_state_lower:
                score -= 0.05  # Clicked something that doesn't exist
    
    # Check for error states
    error_indicators = ['error', 'invalid', 'failed', 'not found', 'unavailable']
    if any(ind in next_state_lower for ind in error_indicators):
        score -= 0.2
    
    # Ensure score stays in valid range
    score = max(0.0, min(1.0, score))
    
    return score