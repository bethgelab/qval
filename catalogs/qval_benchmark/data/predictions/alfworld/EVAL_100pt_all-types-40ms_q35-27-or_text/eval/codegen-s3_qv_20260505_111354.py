import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimate Q-value for ALFWorld environment based on state, action, and next_state.
    Returns a float representing the expected return from taking action a in state s.
    """
    q_value = 0.0
    
    # Check for task completion in next_state
    success_patterns = [
        r'goal\s*is\s*achieved',
        r'task\s*completed',
        r'successful',
        r'goal\s*condition\s*met',
        r'goal\s*is\s*complete',
        r'you\s*have\s*completed',
        r'episode\s*terminated',
        r'goal\s*achieved'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            # Task completed - high Q-value
            q_value = 1.0
            break
    
    if q_value == 1.0:
        return q_value
    
    # Check for failure/error indicators
    failure_patterns = [
        r'cannot\s+',
        r'nothing\s+to\s+',
        r'already\s+',
        r'not\s+found',
        r'invalid',
        r'error',
        r'failed',
        r'cannot\s+perform'
    ]
    
    has_failure = False
    for pattern in failure_patterns:
        if re.search(pattern, next_state, re.IGNORECASE):
            has_failure = True
            break
    
    if has_failure:
        q_value = -0.1
        return q_value
    
    # Analyze action types and their typical value
    action_lower = action.lower()
    
    # Navigation actions - moderate value if moving toward goal
    nav_patterns = [
        r'go\s+to\s+',
        r'walk\s+to\s+',
        r'move\s+to\s+',
        r'go\s+to\s+the\s+'
    ]
    is_navigation = any(re.search(p, action_lower) for p in nav_patterns)
    
    # Object manipulation actions - higher value if productive
    manipulation_patterns = [
        r'take\s+',
        r'pick\s+up\s+',
        r'put\s+in\s+',
        r'put\s+on\s+',
        r'put\s+on\s+the\s+',
        r'clean\s+',
        r'heat\s+',
        r'cool\s+',
        r'fill\s+with\s+',
        r'empty\s+',
        r'turn\s+on\s+',
        r'turn\s+off\s+',
        r'open\s+',
        r'close\s+'
    ]
    is_manipulation = any(re.search(p, action_lower) for p in manipulation_patterns)
    
    # Look for progress indicators in next_state
    progress_patterns = [
        r'you\s+have\s+the',
        r'you\s+took\s+',
        r'you\s+picked\s+up',
        r'you\s+put\s+the',
        r'you\s+opened',
        r'you\s+closed',
        r'you\s+turned\s+on',
        r'you\s+turned\s+off',
        r'you\s+cleaned',
        r'you\s+heated',
        r'you\s+cooled',
        r'you\s+filled',
        r'you\s+emptied',
        r'you\s+went\s+to',
        r'you\s+walked\s+to',
        r'you\s+moved\s+to'
    ]
    
    has_progress = any(re.search(p, next_state, re.IGNORECASE) for p in progress_patterns)
    
    # Check if object is in target location
    target_patterns = [
        r'is\s+in\s+the\s+',
        r'is\s+on\s+the\s+',
        r'is\s+at\s+the\s+'
    ]
    
    # Check for goal-related keywords in state
    goal_keywords = [
        'goal', 'task', 'need', 'must', 'should'
    ]
    
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    
    # Count goal-related mentions
    goal_mentions_state = sum(1 for kw in goal_keywords if kw in state_lower)
    goal_mentions_next = sum(1 for kw in goal_keywords if kw in next_state_lower)
    
    # Estimate Q-value based on features
    if has_progress:
        # Action was productive
        if is_manipulation:
            # Manipulation with progress is typically high value
            q_value = 0.6
        elif is_navigation:
            # Navigation with progress
            q_value = 0.4
        else:
            q_value = 0.3
    else:
        # No clear progress
        if is_manipulation:
            q_value = 0.2
        elif is_navigation:
            q_value = 0.15
        else:
            q_value = 0.1
    
    # Bonus for actions that seem to move toward goal completion
    # Check if next_state shows object in expected location
    if 'in the' in next_state_lower or 'on the' in next_state_lower:
        q_value = min(q_value + 0.1, 0.9)
    
    # Penalize if state complexity increased without clear progress
    if len(next_state) > len(state) * 1.5 and not has_progress:
        q_value = max(q_value - 0.1, -0.2)
    
    # Ensure Q-value is in reasonable range
    q_value = max(-0.5, min(0.95, q_value))
    
    return q_value