import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """Estimate Q-value based on state analysis without lookahead."""
    
    # Normalize text for analysis
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # Check if goal appears achieved (binary reward = 1.0)
    goal_indicators = [
        'success', 'completed', 'done', 'saved', 'sent', 'added',
        'created', 'confirmed', 'task complete', 'message sent',
        'event added', 'item added', 'form submitted'
    ]
    for indicator in goal_indicators:
        if indicator in next_state_lower:
            return 1.0
    
    # Check if state shows goal achievement
    for indicator in goal_indicators:
        if indicator in state_lower:
            return 0.95
    
    # Analyze action type and its typical contribution to progress
    action_score = 0.0
    
    # Click actions on interactive elements
    if 'click' in action_lower:
        # Clicking relevant elements (links, buttons) is generally good
        if any(x in action_lower for x in ['button', 'link', 'submit', 'send', 'add', 'create']):
            action_score = 0.6
        elif 'bid' in action_lower:
            action_score = 0.4  # Generic click
        else:
            action_score = 0.3
    
    # Fill actions (form filling)
    elif 'fill' in action_lower:
        # Filling forms is progress toward submission
        action_score = 0.5
    
    # Press actions (keyboard)
    elif 'press' in action_lower:
        if 'enter' in action_lower:
            action_score = 0.5  # Submitting forms
        else:
            action_score = 0.3
    
    # Scroll actions
    elif 'scroll' in action_lower:
        action_score = 0.2  # Navigation, less direct progress
    
    # Noop
    elif 'noop' in action_lower:
        action_score = 0.1  # Minimal progress
    
    # Check if next state shows progress indicators
    progress_indicators = [
        'loading', 'processing', 'validating', 'saving',
        'new', 'added', 'updated', 'changed'
    ]
    progress_score = 0.0
    for indicator in progress_indicators:
        if indicator in next_state_lower:
            progress_score = max(progress_score, 0.2)
    
    # Check for navigation progress (new page sections)
    navigation_progress = 0.0
    if 'navigate' in next_state_lower or 'page' in next_state_lower:
        navigation_progress = 0.15
    
    # Penalize actions that seem to go backward
    backward_indicators = ['back', 'cancel', 'close', 'delete', 'remove']
    backward_score = 0.0
    for indicator in backward_indicators:
        if indicator in action_lower:
            backward_score = -0.1
            break
    
    # Combine scores with efficiency consideration
    # Base Q-value from action quality
    base_q = action_score + progress_score + navigation_progress + backward_score
    
    # Clip to reasonable range [0, 1]
    q_value = max(0.0, min(1.0, base_q))
    
    # Slight bonus if next state appears more complete than current state
    # (heuristic based on element count or complexity)
    current_elements = len(re.findall(r'\[bid', state_lower))
    next_elements = len(re.findall(r'\[bid', next_state_lower))
    if next_elements > current_elements:
        q_value = min(1.0, q_value + 0.05)
    
    return q_value