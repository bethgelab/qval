def signal_function(state: str) -> float:
    state_lower = state.lower()
    
    # High-value indicators: goal completion, success states
    completion_keywords = ['completed', 'success', 'done', 'saved', 'created', 'added', 
                          'message sent', 'event created', 'todo added', 'goal achieved']
    completion_count = sum(1 for kw in completion_keywords if kw in state_lower)
    
    # Moderate-value indicators: being on relevant pages with interactive elements
    progress_keywords = ['page', 'form', 'input', 'button', 'link', 'menu', 'dropdown',
                        'calendar', 'messenger', 'todo', 'maps', 'editor', 'app']
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    
    # Negative-value indicators: errors, dead ends, no progress
    error_keywords = ['error', 'failed', 'not found', 'invalid', 'missing', 'broken',
                     'unable', 'cannot', 'blocked', 'locked', 'timeout', 'limit exceeded']
    error_count = sum(1 for kw in error_keywords if kw in state_lower)
    
    # Neutral indicators: step count if present (extract numbers)
    step_pattern = re.search(r'(\d+)\s*step(s?)', state_lower)
    steps_taken = 0
    if step_pattern:
        steps_taken = int(step_pattern.group(1))
    
    # Calculate base value from completion signals
    if completion_count > 0:
        return min(1.0, 0.9 + 0.1 * completion_count)
    
    # Check for errors
    if error_count > 2:
        return 0.0
    elif error_count == 1:
        return 0.2
    
    # Check for progress indicators
    if progress_count >= 5:
        base_value = 0.7
    elif progress_count >= 3:
        base_value = 0.5
    elif progress_count >= 1:
        base_value = 0.3
    else:
        base_value = 0.1
    
    # Penalize for high step count (closer to step limit of 45)
    if steps_taken > 30:
        base_value *= 0.5
    elif steps_taken > 20:
        base_value *= 0.75
    elif steps_taken > 10:
        base_value *= 0.9
    
    # Check for goal-related page indicators
    goal_pages = ['calendar', 'messenger', 'maps', 'todo', 'code editor', 'editor']
    goal_page_match = any(page in state_lower for page in goal_pages)
    if goal_page_match:
        base_value = min(1.0, base_value + 0.2)
    
    return max(0.0, min(1.0, base_value))