def signal_function(state: str) -> float:
    """
    Estimate state-value for OpenApps environment.
    Returns a float between 0 and 1 representing estimated probability of task success.
    """
    state_lower = state.lower()
    
    # Check if goal is already achieved (highest value)
    goal_indicators = ['success', 'completed', 'done', 'achieved', 'goal reached', 
                       'task complete', 'finished', 'submitted successfully']
    for indicator in goal_indicators:
        if indicator in state_lower:
            return 1.0
    
    # Check for task completion patterns
    completion_patterns = ['saved', 'sent', 'created', 'added', 'submitted', 
                           'updated', 'confirmed', 'registered']
    completion_count = sum(1 for p in completion_patterns if p in state_lower)
    
    # Check for relevant app context
    app_context_scores = {
        'calendar': 0.4,
        'todo': 0.4,
        'messenger': 0.4,
        'maps': 0.4,
        'editor': 0.4,
        'code': 0.4
    }
    app_score = 0.0
    for app, score in app_context_scores.items():
        if app in state_lower:
            app_score = score
            break
    
    # Check for action elements that indicate progress
    action_elements = ['submit', 'save', 'send', 'create', 'add', 'done', 'finish',
                       'confirm', 'apply', 'go', 'next', 'continue']
    action_count = sum(1 for a in action_elements if a in state_lower)
    
    # Check for form fields being filled or present
    form_indicators = ['input', 'field', 'text', 'date', 'time', 'message',
                       'title', 'description', 'subject', 'content', 'body']
    form_count = sum(1 for f in form_indicators if f in state_lower)
    
    # Check for step progress (if available in state) - simple pattern matching
    step_progress = 0.0
    if 'step' in state_lower:
        parts = state_lower.split('step')
        if len(parts) > 1:
            for part in parts[1:]:
                if 'of' in part or '/' in part:
                    nums = [c for c in part if c.isdigit()]
                    if len(nums) >= 2:
                        try:
                            current = int(nums[0])
                            total = int(nums[1])
                            step_progress = current / total if total > 0 else 0.0
                            break
                        except:
                            pass
    
    # Check for error or failure indicators (reduce value)
    error_indicators = ['error', 'failed', 'invalid', 'missing', 'required',
                        'cannot', 'unable', 'wrong', 'incorrect']
    error_count = sum(1 for e in error_indicators if e in state_lower)
    
    # Check for navigation elements (indicates we might be on wrong page)
    nav_indicators = ['home', 'back', 'cancel', 'exit', 'close']
    nav_count = sum(1 for n in nav_indicators if n in state_lower)
    
    # Calculate base score
    base_score = 0.1
    
    # Goal achievement gives maximum value
    if completion_count > 0:
        base_score = 0.7 + min(completion_count * 0.1, 0.3)
    
    # Add app context score
    base_score += app_score * 0.2
    
    # Add action availability score
    base_score += min(action_count * 0.08, 0.3)
    
    # Add form progress score
    base_score += min(form_count * 0.03, 0.2)
    
    # Add step progress score
    base_score += step_progress * 0.3
    
    # Penalize for errors
    base_score -= error_count * 0.15
    
    # Slight penalty for being on navigation-heavy pages
    base_score -= nav_count * 0.05
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, base_score))