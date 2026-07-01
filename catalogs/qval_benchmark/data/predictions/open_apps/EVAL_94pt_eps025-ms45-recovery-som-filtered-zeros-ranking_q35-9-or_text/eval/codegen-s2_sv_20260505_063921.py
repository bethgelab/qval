def signal_function(state: str) -> float:
    import re
    from collections import Counter
    
    # Extract bid numbers to understand UI structure
    bid_matches = re.findall(r'bid_\d+', state)
    
    # Check for goal completion indicators
    completion_indicators = [
        'saved', 'sent', 'created', 'added', 'completed',
        'success', 'done', 'finished', 'achieved',
        'goal', 'target', 'reach'
    ]
    
    completion_count = sum(1 for indicator in completion_indicators if indicator.lower() in state.lower())
    
    # Check for form fields and inputs
    input_count = len(re.findall(r'input|text|textarea|checkbox|radio', state, re.IGNORECASE))
    
    # Check for navigation elements
    nav_count = len(re.findall(r'nav|link|button|menu', state, re.IGNORECASE))
    
    # Check for progress indicators
    progress_indicators = [
        'progress', 'step', 'current', 'remaining',
        'percentage', 'percent', 'complete'
    ]
    
    progress_count = sum(1 for indicator in progress_indicators if indicator.lower() in state.lower())
    
    # Calculate heuristic value
    base_value = 0.0
    
    # Completion indicators boost value
    base_value += min(completion_count * 0.1, 0.3)
    
    # Progress indicators boost value
    base_value += min(progress_count * 0.15, 0.25)
    
    # Fewer inputs means less work ahead (positive for value)
    input_penalty = min(input_count * 0.05, 0.15)
    base_value += (0.5 - input_penalty)
    
    # Navigation options add flexibility
    nav_bonus = min(nav_count * 0.03, 0.1)
    base_value += nav_bonus
    
    # Ensure value is between 0 and 1
    return max(0.0, min(1.0, base_value))