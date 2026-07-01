def signal_function(state: str):
    import re
    
    progress_score = 0.0
    complexity_penalty = 0.0
    safety_score = 0.0
    goal_proximity = 0.0
    
    # Check for goal completion indicators (positive signals)
    goal_keywords = ['success', 'completed', 'created', 'saved', 'sent', 'added', 'done', 'achieved', 'goal']
    goal_keywords_upper = [kw.upper() for kw in goal_keywords]
    goal_keywords_mixed = [kw.title() for kw in goal_keywords]
    
    for keyword in goal_keywords + goal_keywords_upper + goal_keywords_mixed:
        if keyword.lower() in state.lower():
            progress_score += 0.3
    
    # Check for error indicators (negative signals)
    error_keywords = ['error', 'failed', 'invalid', 'missing', 'not found', 'unavailable', 'blocked']
    for keyword in error_keywords:
        if keyword.lower() in state.lower():
            safety_score -= 0.15
    
    # Estimate complexity by counting interactive elements (bids)
    bid_count = len(re.findall(r'\bid\d+\b', state))
    if bid_count > 10:
        complexity_penalty = 0.2
    elif bid_count > 5:
        complexity_penalty = 0.1
    elif bid_count > 0:
        complexity_penalty = 0.05
    
    # Check for page indicators suggesting goal page
    goal_page_indicators = ['calendar', 'message', 'todo', 'map', 'code', 'editor', 'event', 'task']
    for indicator in goal_page_indicators:
        if indicator.lower() in state.lower():
            goal_proximity += 0.15
    
    # Check for form completion indicators
    form_complete_keywords = ['field', 'input', 'text', 'button', 'submit', 'click']
    form_count = sum(1 for kw in form_complete_keywords if kw.lower() in state.lower())
    if form_count >= 3:
        progress_score += 0.15
    elif form_count >= 1:
        progress_score += 0.05
    
    # Check for navigation progress
    nav_keywords = ['navigate', 'page', 'view', 'tab', 'section', 'heading']
    nav_count = sum(1 for kw in nav_keywords if kw.lower() in state.lower())
    if nav_count >= 2:
        goal_proximity += 0.1
    
    # Calculate total value
    total = progress_score - complexity_penalty + safety_score + goal_proximity
    
    # Clamp to reasonable range [0, 1]
    total = max(0.0, min(1.0, total))
    
    return total, {
        "progress_score": progress_score,
        "complexity_penalty": complexity_penalty,
        "safety_score": safety_score,
        "goal_proximity": goal_proximity,
        "bid_count": bid_count,
    }