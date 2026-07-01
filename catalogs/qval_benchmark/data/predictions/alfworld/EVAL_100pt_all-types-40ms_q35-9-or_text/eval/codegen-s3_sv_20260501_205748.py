def signal_function(state: str) -> float:
    import re
    
    remaining_steps = 40.0
    
    step_match = re.search(r'step\s*[:\s]*(\d+)', state, re.IGNORECASE)
    if step_match:
        current_step = int(step_match.group(1))
        remaining_steps = max(0.0, 40.0 - current_step)
    
    completed = any(kw in state.lower() for kw in ['done', 'complete', 'success', 'finished', 'goal reached', 'task complete'])
    failed = any(kw in state.lower() for kw in ['failed', 'error', 'impossible', 'cannot', 'unable', 'blocked'])
    
    if completed:
        return 1.0
    if failed:
        return 0.0
    
    positive_keywords = ['found', 'picked up', 'moved', 'placed', 'cleaned', 'opened', 'closed', 'reached', 'placed on', 'put in', 'brought', 'carried']
    negative_keywords = ['blocked', 'locked', 'missing', 'broken', 'cannot', 'unable', 'out of reach', 'not here', 'not found']
    
    positive_count = sum(1 for kw in positive_keywords if kw in state.lower())
    negative_count = sum(1 for kw in negative_keywords if kw in state.lower())
    
    progress_ratio = remaining_steps / 40.0
    
    base_value = progress_ratio
    adjustment = (positive_count - negative_count) * 0.05
    
    value = base_value + adjustment
    value = max(0.0, min(1.0, value))
    
    return value