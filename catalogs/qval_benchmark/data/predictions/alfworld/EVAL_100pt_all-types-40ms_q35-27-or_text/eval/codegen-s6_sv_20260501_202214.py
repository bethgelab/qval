import re
import math

def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check if task is completed
    completion_keywords = ['success', 'completed', 'done', 'finished', 'achieved']
    for keyword in completion_keywords:
        if keyword in state_lower:
            return 1.0
    
    # Check for failure indicators
    failure_keywords = ['failed', 'error', 'cannot', 'invalid', 'timeout', 'step limit']
    for keyword in failure_keywords:
        if keyword in state_lower:
            return 0.0
    
    # Count completed actions/subtasks
    completed_count = len(re.findall(r'(?:done|completed|achieved|success)', state_lower))
    
    # Estimate task complexity by counting action keywords
    action_keywords = ['put', 'move', 'take', 'clean', 'open', 'close', 'heat', 'cool', 'examine', 'find', 'go', 'navigate']
    complexity = len(re.findall(r'\b(?:' + '|'.join(action_keywords) + r')\b', state_lower))
    
    # Look for goal/task description to estimate total subtasks
    goal_match = re.search(r'(?:goal|task)[:\s]+(.+?)(?:\.|$)', state_lower)
    if goal_match:
        goal_text = goal_match.group(1)
        # Count objects/actions mentioned in goal
        goal_items = len(re.findall(r'\b(?:' + '|'.join(action_keywords) + r')\b', goal_text))
        if goal_items > 0:
            complexity = max(complexity, goal_items)
    
    # Calculate progress ratio
    if complexity > 0:
        progress = min(1.0, completed_count / complexity)
    else:
        progress = 0.3
    
    # Estimate remaining steps based on progress
    # Assume optimal path takes roughly complexity * 2 steps on average
    estimated_remaining = max(0, complexity * 2 * (1 - progress))
    
    # Apply exponential discount based on remaining steps
    # Higher discount for states further from completion
    discount = math.exp(-estimated_remaining / 15)
    
    # Base value from progress (0.8 max for incomplete states)
    base_value = progress * 0.8
    
    # Combine progress and efficiency
    value = base_value * discount
    
    # Add small bonus for being in a navigable state
    if 'you are' in state_lower or 'location' in state_lower:
        value = min(1.0, value + 0.05)
    
    # Ensure value is in valid range
    return max(0.0, min(1.0, value))