def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    score = 0.0
    
    # Check if goal was achieved (binary reward = 1.0)
    goal_indicators = [
        'completed', 'success', 'done', 'submitted', 'added',
        'created', 'saved', 'sent', 'updated', 'confirmed',
        'event added', 'message sent', 'task completed',
        'calendar event', 'todo item'
    ]
    
    next_lower = next_state.lower()
    for indicator in goal_indicators:
        if indicator in next_lower:
            return 1.0
    
    # Extract action type
    action_match = re.match(r"(\w+)\(", action)
    action_type = action_match.group(1) if action_match else "noop"
    
    # Progress indicators from state transition
    progress_score = 0.0
    
    # State changed indicates action had effect
    if next_state.strip() != state.strip():
        progress_score += 0.25
    
    # Appropriate action types for web navigation
    productive_actions = ['click', 'fill', 'press']
    if action_type in productive_actions:
        progress_score += 0.2
    
    # Check for error/failure states
    error_indicators = ['error', 'invalid', 'failed', 'not found', 'missing', 'unable']
    for indicator in error_indicators:
        if indicator in next_lower:
            progress_score -= 0.3
    
    # Navigation progress - moving through forms/pages
    nav_indicators = ['form', 'dialog', 'modal', 'page', 'screen', 'input', 'button']
    nav_count = sum(1 for ind in nav_indicators if ind in next_lower)
    progress_score += min(nav_count * 0.08, 0.25)
    
    # Action-specific bonuses
    if action_type == 'fill' and 'input' in next_lower:
        progress_score += 0.15
    
    if action_type == 'click' and ('button' in next_lower or 'submit' in next_lower):
        progress_score += 0.15
    
    # Check for goal-related keywords (not yet achieved)
    goal_keywords = ['todo', 'calendar', 'message', 'event', 'task', 'add', 'create']
    goal_matches = sum(1 for kw in goal_keywords if kw in next_lower)
    progress_score += min(goal_matches * 0.05, 0.1)
    
    # Efficiency consideration - simpler states may indicate progress
    state_token_count = len(next_state.split())
    if state_token_count < 80:
        progress_score += 0.1
    
    # Base Q-value with progress bonus
    base_q = 0.4
    
    # Cap and floor the score
    final_score = base_q + progress_score
    return max(0.0, min(1.0, final_score))