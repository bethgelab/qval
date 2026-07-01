def signal_function(state: str, action: str, next_state: str):
    # Parse action type and parameters
    action_lower = action.lower()
    action_type = None
    action_target = ""
    if action_lower.startswith('click'):
        action_type = 'click'
        action_target = action_lower[6:].strip()
    elif action_lower.startswith('fill'):
        action_type = 'fill'
        action_target = action_lower[5:].strip()
    elif action_lower.startswith('press'):
        action_type = 'press'
        action_target = action_lower[6:].strip()
    elif action_lower.startswith('noop'):
        action_type = 'noop'
    elif action_lower.startswith('scroll'):
        action_type = 'scroll'
    
    # Detect app context from state
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    app_context = 'unknown'
    if 'todo' in state_lower or 'task' in state_lower:
        app_context = 'todo'
    elif 'calendar' in state_lower or 'event' in state_lower:
        app_context = 'calendar'
    elif 'message' in state_lower or 'chat' in state_lower or 'send' in state_lower:
        app_context = 'messenger'
    elif 'map' in state_lower or 'route' in state_lower or 'location' in state_lower:
        app_context = 'maps'
    elif 'code' in state_lower or 'editor' in state_lower:
        app_context = 'code'
    
    # State change detection
    state_changed = state != next_state
    
    # Critical action keywords that indicate progress opportunities
    critical_action_keywords = ['submit', 'send', 'save', 'add', 'create', 'confirm', 'ok', 'done', 'next', 'finish', 'complete']
    
    # Success probability estimation - core Q-value component
    success_probability = 0.0
    
    # Strong completion indicators - task is definitively complete
    strong_keywords = ['completed successfully', 'saved successfully', 'sent successfully', 
                       'event created successfully', 'task completed successfully', 
                       'submitted successfully', 'created successfully', 'success',
                       'completed!', 'saved!', 'sent!', 'done!', 'finished!', 'goal achieved']
    strong_count = sum(1 for kw in strong_keywords if kw in next_state_lower)
    if strong_count > 0:
        success_probability = 1.0
    
    # Medium completion indicators - task-specific success messages
    medium_keywords = ['added', 'created', 'saved', 'sent', 'completed', 'updated', 
                       'submitted', 'finished', 'done', 'found', 'results', 'scheduled']
    medium_count = sum(1 for kw in medium_keywords if kw in next_state_lower)
    if medium_count > 0 and success_probability < 1.0:
        success_probability = 0.75
    
    # App-specific success patterns with calibrated confidence
    if app_context == 'calendar' and any(kw in next_state_lower for kw in ['event created', 'event added', 'appointment added', 'meeting scheduled']):
        success_probability = max(success_probability, 0.80)
    if app_context == 'messenger' and any(kw in next_state_lower for kw in ['message sent', 'sent', 'delivered', 'message delivered']):
        success_probability = max(success_probability, 0.78)
    if app_context == 'todo' and any(kw in next_state_lower for kw in ['task added', 'todo created', 'item added', 'task created']):
        success_probability = max(success_probability, 0.78)
    if app_context == 'maps' and any(kw in next_state_lower for kw in ['route found', 'directions found', 'navigation started', 'search results']):
        success_probability = max(success_probability, 0.72)
    
    # Progress indicators - linear scaling with task completion
    progress_score = 0.0
    
    # Weak progress indicators (app-specific context)
    weak_keywords = []
    if app_context == 'todo':
        weak_keywords = ['todo', 'task', 'item', 'checklist', 'pending', 'list', 'complete']
    elif app_context == 'calendar':
        weak_keywords = ['event', 'meeting', 'appointment', 'calendar', 'schedule', 'date', 'time', 'month']
    elif app_context == 'messenger':
        weak_keywords = ['message', 'chat', 'conversation', 'inbox', 'contact', 'recipient', 'compose']
    elif app_context == 'maps':
        weak_keywords = ['route', 'direction', 'location', 'map', 'navigation', 'address', 'search', 'place']
    elif app_context == 'code':
        weak_keywords = ['code', 'editor', 'file', 'function', 'snippet', 'script', 'run']
    
    # Linear scaling: each keyword contributes proportionally, capped at 0.5
    weak_count = sum(1 for kw in weak_keywords if kw in next_state_lower)
    progress_score += min(weak_count * 0.06, 0.5)
    
    # Form interaction progress - higher weight for meaningful interactions
    if action_type == 'fill' and state_changed:
        progress_score += 0.20
    if action_type == 'click' and state_changed:
        progress_score += 0.15
    if action_type == 'press' and state_changed:
        progress_score += 0.18
    
    # Scroll-specific: bonus for revealing critical action buttons
    if action_type == 'scroll' and state_changed:
        if any(kw in next_state_lower for kw in critical_action_keywords):
            progress_score += 0.20
        else:
            progress_score += 0.08
    
    # Form completion patterns (multiple fields likely filled)
    if action_type == 'fill':
        field_indicators = ['input', 'text', 'field', 'name', 'email', 'phone', 'address', 'title', 'description', 'subject', 'body']
        field_count = sum(1 for kw in field_indicators if kw in next_state_lower)
        if field_count >= 3:
            progress_score += 0.18
        elif field_count == 2:
            progress_score += 0.10
        elif field_count == 1:
            progress_score += 0.05
    
    # State change bonus - indicates we're making progress
    state_change_bonus = 0.0
    if state_changed:
        critical_appeared = any(kw in next_state_lower for kw in critical_action_keywords)
        if critical_appeared:
            state_change_bonus = 0.18
        else:
            state_change_bonus = 0.10
    else:
        state_change_bonus = -0.08
    
    # Action quality baseline - meaningful actions get higher base value
    action_quality = 0.0
    if action_type == 'click':
        action_quality = 0.10
    elif action_type == 'fill':
        action_quality = 0.15
    elif action_type == 'press':
        action_quality = 0.08
    elif action_type == 'noop':
        action_quality = -0.20
    elif action_type == 'scroll':
        action_quality = 0.05
    
    # Action specificity bonus - reward targeted actions
    action_specificity = 0.0
    if action_type in ['click', 'fill', 'press'] and action_target and len(action_target) > 0:
        action_specificity = 0.08
    if action_type == 'click' and any(kw in action_target for kw in ['submit', 'send', 'save', 'add', 'create', 'confirm']):
        action_specificity = 0.12
    
    # Efficiency bonus - reward faster progress (45-step limit)
    efficiency_bonus = 0.0
    if state_changed and success_probability < 1.0:
        # Estimate remaining steps based on progress
        if success_probability > 0.7:
            efficiency_bonus = 0.12
        elif success_probability > 0.4:
            efficiency_bonus = 0.08
        elif success_probability > 0.2:
            efficiency_bonus = 0.04
        else:
            efficiency_bonus = 0.0
    
    # Penalty for state changes without meaningful progress (inefficient)
    if state_changed and success_probability < 0.2 and progress_score < 0.10:
        efficiency_bonus -= 0.10
    
    # Safety penalties - reduced when there's clear progress
    safety_penalty = 0.0
    
    # Hard penalties for clearly unproductive actions (only when no progress)
    if action_type == 'noop' and not state_changed and success_probability < 0.5:
        safety_penalty = -0.30
    
    if action_type == 'scroll' and not state_changed and success_probability < 0.5:
        safety_penalty = -0.18
    
    # Moderate penalty for state changes without progress indicators
    if state_changed and success_probability < 0.2 and progress_score < 0.10:
        safety_penalty = -0.12
    
    # Small penalty for repeated action types without progress
    if action_type == 'click' and not state_changed and success_probability < 0.5:
        safety_penalty = -0.10
    
    # Penalty for fill actions without state change
    if action_type == 'fill' and not state_changed:
        safety_penalty = -0.15
    
    # Reward for strong progress signals - reduces penalties
    if strong_count > 0:
        safety_penalty = max(safety_penalty, 0.0)
    
    # Calculate total Q-value estimate
    total = success_probability + progress_score + action_quality + state_change_bonus + action_specificity + efficiency_bonus + safety_penalty
    
    # Clamp to [0, 1] to represent probability of eventual success
    total = max(0.0, min(total, 1.0))
    
    return total, {
        "success_probability": success_probability,
        "progress_score": progress_score,
        "action_quality": action_quality,
        "state_change_bonus": state_change_bonus,
        "action_specificity": action_specificity,
        "efficiency_bonus": efficiency_bonus,
        "safety_penalty": safety_penalty,
    }