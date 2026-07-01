import re
import math

def signal_function(state: str) -> float:
    value = 0.5
    
    # Check for task completion indicators
    completion_keywords = ['completed', 'done', 'success', 'goal', 'achieved', 'target']
    for keyword in completion_keywords:
        if keyword in state.lower():
            value = 0.95
            break
    
    # Check for error/failure indicators
    error_keywords = ['error', 'failed', 'invalid', 'wrong', 'not found', 'missing', 'unavailable']
    for keyword in error_keywords:
        if keyword in state.lower():
            value = 0.1
            break
    
    # Check for page context indicators (closer to goal pages)
    if 'calendar' in state.lower():
        value = max(value, 0.65)
    elif 'todo' in state.lower():
        value = max(value, 0.65)
    elif 'messenger' in state.lower():
        value = max(value, 0.65)
    elif 'maps' in state.lower():
        value = max(value, 0.65)
    elif 'code' in state.lower():
        value = max(value, 0.65)
    
    # Check for form interaction indicators (progress in task)
    if 'fill' in state.lower() or 'input' in state.lower() or 'value=' in state.lower():
        value = max(value, 0.55)
    
    # Check for button/action indicators (potential to complete task)
    if 'button' in state.lower() or 'click' in state.lower() or 'submit' in state.lower():
        value = max(value, 0.55)
    
    # Check for element count (more elements = more context/progress)
    bid_count = len(re.findall(r'bid\d+', state))
    if bid_count > 15:
        value = max(value, 0.6)
    elif bid_count < 3:
        value = max(value, 0.35)
    
    # Check for scroll/navigation indicators
    if 'scroll' in state.lower() or 'navigate' in state.lower():
        value = max(value, 0.5)
    
    # Check for time/step proximity indicators
    if 'step' in state.lower():
        match = re.search(r'\d+', state)
        if match:
            steps = int(match.group())
            if steps <= 10:
                value = max(value, 0.7)
            elif steps <= 25:
                value = max(value, 0.55)
            else:
                value = max(value, 0.45)
    
    # Check for task-specific progress indicators
    progress_keywords = ['progress', 'remaining', 'current', 'active', 'selected', 'highlighted']
    for keyword in progress_keywords:
        if keyword in state.lower():
            value = max(value, 0.5)
            break
    
    # Check for goal-related elements
    goal_keywords = ['add', 'create', 'send', 'save', 'update', 'edit', 'delete']
    for keyword in goal_keywords:
        if keyword in state.lower():
            value = max(value, 0.55)
            break
    
    # Ensure value is in valid range [0, 1]
    value = max(0.0, min(1.0, value))
    
    return value