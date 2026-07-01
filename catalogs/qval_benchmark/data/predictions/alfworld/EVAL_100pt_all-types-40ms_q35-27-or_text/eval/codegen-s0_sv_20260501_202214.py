def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # Check for task completion - highest value
    completion_indicators = ['done', 'success', 'completed', 'goal achieved', 'task completed', 'you have successfully', 'congratulations']
    for indicator in completion_indicators:
        if indicator in state_lower:
            return 1.0
    
    # Initialize probability estimate
    prob = 0.0
    
    # Feature 1: Has the target object (strong indicator of progress)
    has_object = any(x in state_lower for x in ['have ', 'holding ', 'carrying ', 'in hand', 'with me'])
    if has_object:
        prob += 0.35
    
    # Feature 2: Object in correct state (important for many tasks)
    good_state = any(x in state_lower for x in ['clean', 'open', 'unlocked', 'on ', 'turned on'])
    if good_state:
        prob += 0.25
    
    # Feature 3: In relevant location
    in_room = any(x in state_lower for x in ['kitchen', 'bedroom', 'bathroom', 'livingroom', 'diningroom', 'office', 'room'])
    if in_room:
        prob += 0.15
    
    # Feature 4: Goal awareness mentioned in state
    goal_mentioned = any(x in state_lower for x in ['goal', 'target', 'need', 'want', 'task'])
    if goal_mentioned:
        prob += 0.1
    
    # Feature 5: Multiple positive features boost confidence
    positive_features = sum([has_object, good_state, in_room])
    if positive_features >= 3:
        prob += 0.15
    elif positive_features >= 2:
        prob += 0.1
    
    # Feature 6: Negative indicators reduce probability
    negative_indicators = ['dirty', 'closed', 'locked', 'broken', 'cannot', 'fail', 'error']
    has_negative = any(x in state_lower for x in negative_indicators)
    if has_negative:
        prob -= 0.1
    
    # Base probability for being in environment
    prob += 0.05
    
    # Clamp to valid probability range
    return max(0.0, min(1.0, prob))