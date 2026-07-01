import re
from collections import Counter

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for taking action in state resulting in next_state.
    Uses heuristics based on state text analysis to predict progress toward goal.
    """
    # Define progress indicators commonly found in ALFWorld
    progress_keywords = {
        'on table': 0.1,
        'in hand': 0.3,
        'on shelf': 0.1,
        'on floor': 0.1,
        'clean': 0.4,
        'dirty': -0.3,
        'washed': 0.5,
        'dried': 0.4,
        'folded': 0.4,
        'placed': 0.3,
        'picked up': 0.3,
        'put down': 0.2,
        'open': 0.2,
        'closed': 0.2,
        'charged': 0.5,
        'recharged': 0.5,
        'replaced': 0.4,
        'installed': 0.4,
        'fixed': 0.4,
        'repaired': 0.4,
        'assembled': 0.5,
        'unpacked': 0.3,
        'packed': 0.3,
        'delivered': 0.6,
        'delivered to': 0.6,
        'placed in': 0.4,
        'placed on': 0.3,
        'at': 0.2,
        'near': 0.15,
        'in front of': 0.15,
        'behind': 0.1,
        'left of': 0.1,
        'right of': 0.1,
        'under': 0.1,
        'above': 0.1,
    }
    
    # Completion indicators
    completion_keywords = {
        'task completed': 0.95,
        'task done': 0.95,
        'goal achieved': 0.95,
        'successfully': 0.9,
        'done': 0.85,
        'finished': 0.85,
        'completed': 0.9,
        'done!': 0.9,
        'success': 0.95,
        'success!': 0.95,
    }
    
    # Negative indicators
    negative_keywords = {
        'failed': 0.1,
        'failure': 0.1,
        'error': 0.1,
        'cannot': 0.1,
        'unable': 0.1,
        'not possible': 0.1,
        'out of stock': 0.1,
        'broken': 0.2,
        'damaged': 0.2,
        'too heavy': 0.1,
        'too far': 0.1,
        'blocked': 0.1,
        'occupied': 0.1,
        'full': 0.1,
        'empty': 0.05,
    }
    
    # Count occurrences of keywords in next_state
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    action_lower = action.lower()
    
    # Calculate progress score from next_state
    progress_score = 0.0
    for keyword, value in progress_keywords.items():
        if keyword in next_state_lower:
            progress_score += value
    
    # Calculate completion score from next_state
    completion_score = 0.0
    for keyword, value in completion_keywords.items():
        if keyword in next_state_lower:
            completion_score += value
    
    # Calculate negative score from next_state
    negative_score = 0.0
    for keyword, value in negative_keywords.items():
        if keyword in next_state_lower:
            negative_score += abs(value)
    
    # Base Q-value from next_state analysis
    base_value = completion_score + (progress_score * 0.3) - (negative_score * 0.3)
    
    # Check if action makes sense (simple heuristic)
    action_type = action_lower.split()
    if len(action_type) >= 2:
        verb = action_type[0]
        if verb in ['go', 'move', 'walk', 'navigate', 'travel']:
            # Navigation action
            if 'to' in action_lower or 'to the' in action_lower:
                base_value += 0.1
            elif 'to' not in action_lower and 'the' not in action_lower:
                base_value -= 0.05
        elif verb in ['pick', 'take', 'grab', 'hold']:
            # Pick up action
            if 'up' in action_lower or 'up ' in action_lower:
                base_value += 0.1
        elif verb in ['put', 'place', 'drop', 'leave']:
            # Put down action
            if 'down' in action_lower or 'on' in action_lower or 'in' in action_lower:
                base_value += 0.08
        elif verb in ['clean', 'wash', 'dry', 'fold', 'iron']:
            # Manipulation action
            base_value += 0.05
    
    # Check for state changes that indicate progress
    state_changes = 0
    if next_state_lower != state_lower:
        # Check for object position changes
        if 'in hand' in next_state_lower and 'in hand' not in state_lower:
            state_changes += 1
        if 'on table' in next_state_lower and 'on table' not in state_lower:
            state_changes += 0.5
        if 'clean' in next_state_lower and 'clean' not in state_lower:
            state_changes += 0.5
        if 'dirty' in state_lower and 'dirty' not in next_state_lower:
            state_changes += 0.5
    
    # Adjust based on state changes
    if state_changes > 0:
        base_value += state_changes * 0.05
    
    # Clamp value to reasonable range
    base_value = max(0.0, min(1.0, base_value))
    
    # Add small bonus for action that seems purposeful
    if action_lower and len(action_lower) > 5:
        base_value += 0.02
    
    return round(base_value, 3)