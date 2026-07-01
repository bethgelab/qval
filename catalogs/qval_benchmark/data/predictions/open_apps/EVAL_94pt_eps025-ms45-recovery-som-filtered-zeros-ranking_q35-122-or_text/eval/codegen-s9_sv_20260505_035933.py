def signal_function(state: str) -> float:
    import re
    import math
    
    if not state or not isinstance(state, str):
        return 0.0
    
    state_lower = state.lower()
    
    # Check if task is already complete
    completion_indicators = [
        'completed', 'success', 'done', 'saved', 'added', 'created',
        'sent', 'published', 'task added', 'event created', 'message sent',
        'goal achieved', 'task complete', 'successfully', 'verification passed'
    ]
    
    is_complete = any(indicator in state_lower for indicator in completion_indicators)
    if is_complete:
        return 1.0
    
    # Extract step information if available
    step_match = re.search(r'step[:\s]+(\d+)', state, re.IGNORECASE)
    current_step = int(step_match.group(1)) if step_match else 0
    
    max_steps = 45
    steps_remaining = max(0, max_steps - current_step)
    
    # Discount based on steps remaining - fewer steps means harder to complete
    # Use exponential decay for step penalty
    step_penalty = math.exp(-0.08 * (max_steps - steps_remaining))
    
    # Count interactive elements (bids) - more options = more flexibility
    bid_patterns = [
        r"bid['\"]?\s*[:=]\s*['\"]?(\d+)",
        r'\[bid(\d+)\]',
        r'bid(\d+)',
        r'"bid":\s*(\d+)'
    ]
    
    bid_count = 0
    for pattern in bid_patterns:
        bid_count += len(re.findall(pattern, state))
    
    # Normalize bid count (typical range 5-30 interactive elements)
    bid_score = min(bid_count / 25.0, 1.0) if bid_count > 0 else 0.0
    
    # Check for app-specific progress indicators
    progress_indicators = {
        'todo': ['task', 'todo', 'check', 'complete', 'list', 'item'],
        'calendar': ['event', 'calendar', 'date', 'time', 'meeting', 'schedule'],
        'messenger': ['message', 'send', 'chat', 'contact', 'compose'],
        'maps': ['location', 'map', 'address', 'search', 'destination'],
        'code': ['code', 'editor', 'file', 'save', 'run', 'output']
    }
    
    keyword_score = 0.0
    for app, keywords in progress_indicators.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', state_lower):
                keyword_score += 0.15
                break
    
    keyword_score = min(keyword_score, 1.0)
    
    # Check for error or failure indicators (negative signals)
    error_patterns = ['error', 'failed', 'invalid', 'incorrect', 'cannot', 'unavailable']
    error_count = sum(1 for pattern in error_patterns if pattern in state_lower)
    error_penalty = min(error_count * 0.15, 0.5)
    
    # Check for navigation progress (links, pages visited)
    nav_indicators = ['page', 'view', 'section', 'tab', 'menu', 'navigate']
    nav_score = min(sum(1 for nav in nav_indicators if nav in state_lower) * 0.1, 0.5)
    
    # Combine all signals with appropriate weights
    # Completion is most important, then step efficiency, then element availability
    base_value = 0.0
    
    # Step efficiency contributes significantly
    base_value += 0.35 * step_penalty
    
    # Available actions contribute to success probability
    base_value += 0.25 * bid_score
    
    # Task-relevant content indicates progress
    base_value += 0.25 * keyword_score
    
    # Navigation progress
    base_value += 0.15 * nav_score
    
    # Apply error penalty
    base_value = max(base_value - error_penalty, 0.0)
    
    # Early step bonus if we're just starting
    if current_step <= 3:
        base_value += 0.1
    
    # Late step penalty if running out of time
    if steps_remaining <= 5 and current_step > 10:
        base_value *= 0.7
    
    # Clamp to valid range
    return max(0.0, min(1.0, base_value))