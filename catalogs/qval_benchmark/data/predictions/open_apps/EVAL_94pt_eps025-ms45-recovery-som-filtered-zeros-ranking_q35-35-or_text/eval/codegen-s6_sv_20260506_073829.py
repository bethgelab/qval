def signal_function(state: str) -> float:
    # Check if goal is already achieved (terminal state)
    goal_achievers = ['GOAL_ACHIEVED', 'success', 'completed', 'done', 'task_complete']
    if any(term in state for term in goal_achievers):
        return 1.0
    
    # Check for error/blocker states
    error_terms = ['error', 'blocked', 'failed', 'not_found', 'invalid', 'unavailable']
    if any(term in state.lower() for term in error_terms):
        return 0.0
    
    # Base value from progress indicators
    value = 0.0
    
    # Check for interactive elements that suggest progress
    interactive_elements = state.count('button') + state.count('input') + state.count('link')
    value += min(interactive_elements * 0.02, 0.3)
    
    # Check for form-related progress indicators
    form_terms = ['form', 'field', 'fill', 'enter', 'submit']
    if any(term in state.lower() for term in form_terms):
        value += 0.15
    
    # Check for completion indicators in state
    completion_terms = ['saved', 'added', 'sent', 'created', 'updated']
    if any(term in state.lower() for term in completion_terms):
        value += 0.25
    
    # Check for navigation progress (moving through pages)
    nav_terms = ['page', 'view', 'screen', 'tab']
    if any(term in state.lower() for term in nav_terms):
        value += 0.1
    
    # Check for app context relevance (should be on right app)
    app_terms = ['calendar', 'todo', 'messenger', 'maps', 'editor']
    if any(term in state.lower() for term in app_terms):
        value += 0.15
    
    # Step efficiency bonus (fewer steps used = better)
    # Parse step count from state if available
    step_bonus = 0.0
    step_match = re.search(r'step[:\s]+(\d+)', state, re.IGNORECASE)
    if step_match:
        current_step = int(step_match.group(1))
        step_bonus = max(0, (45 - current_step) / 45 * 0.2)
    else:
        step_bonus = 0.1  # Default bonus if step info not available
    
    value += step_bonus
    
    # Penalty for long paths without progress
    if 'waiting' in state.lower() or 'loading' in state.lower():
        value -= 0.05
    
    # Clamp to [0, 1] range
    value = max(0.0, min(1.0, value))
    
    return value