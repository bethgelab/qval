def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    import math
    
    # Success indicators in next_state
    success_patterns = [
        r'done', r'success', r'goal', r'completed', r'task done',
        r'task completed', r'goal reached', r'object placed',
        r'put on', r'placed on', r'cleaned', r'washed'
    ]
    
    # Failure indicators in state
    failure_patterns = [
        r'error', r'fail', r'cannot', r'not able', r'blocked',
        r'cannot', r'failed', r'error', r'problem', r'issue'
    ]
    
    # Progress indicators
    progress_patterns = [
        r'closer', r'approaching', r'near', r'next', r'current'
    ]
    
    # Check for success in next_state
    next_success = any(re.search(p, next_state, re.IGNORECASE) for p in success_patterns)
    state_failure = any(re.search(p, state, re.IGNORECASE) for p in failure_patterns)
    
    # Check for goal-related keywords
    goal_keywords = ['goal', 'task', 'object', 'clean', 'place', 'put', 'move', 'find', 'bring']
    goal_in_state = any(re.search(r'\b' + kw + r'\b', state, re.IGNORECASE) for kw in goal_keywords)
    goal_in_next = any(re.search(r'\b' + kw + r'\b', next_state, re.IGNORECASE) for kw in goal_keywords)
    
    # Estimate progress improvement
    progress_improvement = 0.0
    if goal_in_next and not goal_in_state:
        progress_improvement = 0.3
    elif goal_in_next and goal_in_state:
        progress_improvement = 0.1
    elif not goal_in_next and goal_in_state:
        progress_improvement = -0.1
    
    # Check if next_state shows more completion than state
    completion_score = 0.0
    if next_success:
        completion_score = 0.8
    elif state_failure:
        completion_score = -0.3
    elif next_success and state_failure:
        completion_score = 0.6
    
    # Base estimate on action relevance
    action_keywords = ['go', 'move', 'take', 'put', 'clean', 'wash', 'find', 'bring', 'open', 'close']
    action_has_keyword = any(re.search(r'\b' + kw + r'\b', action, re.IGNORECASE) for kw in action_keywords)
    
    # Combine signals
    q_value = 0.0
    
    # If we see success in next_state, give high value
    if next_success:
        q_value = 0.9
    elif state_failure:
        q_value = 0.1
    elif action_has_keyword:
        q_value = 0.4 + progress_improvement
    else:
        q_value = 0.2 + progress_improvement
    
    # Clamp between 0 and 1
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value