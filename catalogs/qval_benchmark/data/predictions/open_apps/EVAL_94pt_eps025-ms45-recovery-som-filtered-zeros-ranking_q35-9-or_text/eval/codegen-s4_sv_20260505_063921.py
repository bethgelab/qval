import re

def signal_function(state: str) -> float:
    # Count interactive elements (bid numbers)
    bids = re.findall(r'\bid\d+\b', state)
    bid_count = len(bids)
    
    # Check for goal completion indicators
    completion_keywords = ['completed', 'success', 'saved', 'added', 'created', 'done', 'finish']
    completed = any(kw in state.lower() for kw in completion_keywords)
    
    # Check for error or obstacle indicators
    error_keywords = ['error', 'failed', 'invalid', 'missing', 'required', 'cannot', 'unable']
    has_error = any(kw in state.lower() for kw in error_keywords)
    
    # Check for form elements (indicates task complexity)
    form_elements = re.findall(r'(input|textarea|select|button)', state, re.IGNORECASE)
    form_count = len(form_elements)
    
    # Check for page navigation context
    app_context = re.search(r'(todo|calendar|messenger|maps|editor)', state, re.IGNORECASE)
    is_on_correct_page = app_context is not None
    
    # Calculate base progress estimate
    if completed:
        return 0.95
    
    if has_error:
        return 0.15
    
    # Estimate steps needed based on interactive elements
    # More elements = potentially more steps to complete
    estimated_steps_needed = max(1, min(10, bid_count + form_count))
    
    # Steps remaining from 45 step limit
    steps_remaining = 45 - estimated_steps_needed
    step_factor = max(0.0, min(1.0, steps_remaining / 45))
    
    # Base value from available interactive elements
    # Having more options increases success probability
    element_factor = min(1.0, bid_count / 8.0)
    
    # If on correct page, higher base value
    page_factor = 1.0 if is_on_correct_page else 0.6
    
    # Combine factors
    value = element_factor * page_factor * step_factor
    
    # Ensure value is in valid range
    return max(0.0, min(1.0, value))