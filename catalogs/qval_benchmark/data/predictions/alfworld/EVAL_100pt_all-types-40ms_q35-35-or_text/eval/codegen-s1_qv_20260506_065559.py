import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in ALFWorld.
    Returns a float between 0.0 and 1.0 representing the expected return.
    """
    # Normalize strings for comparison
    state_lower = state.lower()
    action_lower = action.lower()
    next_state_lower = next_state.lower()
    
    # 1. Check for Terminal Success
    # In ALFWorld, successful episodes often contain specific completion phrases
    success_indicators = ['completed', 'success', 'done', 'task complete', 'goal satisfied']
    for indicator in success_indicators:
        if indicator in next_state_lower:
            return 1.0
    
    # 2. Check for Stagnation (Invalid Action or No Progress)
    # If state doesn't change, the action likely failed or was invalid
    if state == next_state:
        return 0.05
    
    # 3. Extract Goal from State
    # ALFWorld states typically start with "Goal: ..."
    goal_match = re.search(r'Goal:\s*(.+)', state)
    goal_text = goal_match.group(1).lower() if goal_match else ""
    
    # 4. Heuristic Progress Scoring
    # We reward actions that align with the goal and states that show progress
    
    # Common goal verbs in ALFWorld tasks
    goal_verbs = ['clean', 'move', 'put', 'heat', 'cool', 'wash', 'take', 'go']
    
    # Check if action is relevant to the goal
    action_relevant = any(verb in action_lower for verb in goal_verbs)
    
    # Check if next_state indicates progress towards goal specifics
    progress_indicators = []
    
    if 'clean' in goal_text or 'wash' in goal_text:
        if 'clean' in next_state_lower or 'wash' in next_state_lower:
            progress_indicators.append('cleaned')
        # Check for state change of objects (e.g., "cloth is clean")
        if 'clean' in next_state_lower:
             progress_indicators.append('state_change')
    
    if 'move' in goal_text or 'put' in goal_text:
        # Look for location indicators in next_state
        location_markers = ['in ', 'on ', 'at ', 'to ']
        if any(marker in next_state_lower for marker in location_markers):
            progress_indicators.append('location_change')
            
    # Calculate score based on alignment
    score = 0.0
    
    # High score if action is relevant AND state shows progress
    if action_relevant and len(progress_indicators) > 0:
        score = 0.8
    # Medium score if action is relevant but state change is ambiguous
    elif action_relevant:
        score = 0.5
    # Low score if state shows progress but action seems irrelevant (e.g. random move)
    elif len(progress_indicators) > 0:
        score = 0.3
    # Very low score for irrelevant action and no progress
    else:
        score = 0.1
        
    # Return the estimated Q-value
    return float(score)