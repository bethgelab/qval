def signal_function(state: str):
    import re
    
    state_lower = state.lower()
    
    # Initialize component values
    goal_success = 0.0
    error_penalty = 0.0
    form_completion = 0.0
    navigation_progress = 0.0
    action_efficiency = 0.0
    wandering_penalty = 0.0
    completion_proximity = 0.0
    turn_efficiency = 0.0
    
    # Check for explicit success/completion indicators - require MULTIPLE for high value
    success_patterns = [
        'task added', 'event created', 'message sent', 'todo added',
        'calendar event', 'form submitted', 'confirmation', 'saved',
        'successfully', 'completed', 'created', 'added', 'sent',
        'todo list updated', 'event saved', 'message delivered',
        'file uploaded', 'document saved', 'location added',
        'successfully created', 'successfully added', 'successfully sent',
        'success', 'done', 'finished', 'confirmed', 'task complete',
        'event complete', 'message complete', 'all tasks completed',
        'event successfully', 'message successfully', 'todo successfully'
    ]
    
    success_count = sum(1 for pattern in success_patterns if pattern in state_lower)
    
    # Only award high goal_success if we have strong evidence of completion
    if success_count >= 3:
        goal_success = 1.0
    elif success_count >= 2:
        goal_success = 0.7
    elif success_count >= 1:
        goal_success = 0.3
    else:
        goal_success = 0.0
    
    # Check for error/failure states
    critical_errors = ['error', 'failed', 'invalid', 'blocked', 'cannot', 'unavailable', 
                       '404', '500', 'unable', 'rejected', 'denied', 'exception', 'crash',
                       'not working', 'refused', 'forbidden', 'page not found', 'server error']
    warning_errors = ['warning', 'alert', 'problem', 'issue', 'missing', 'required', 
                      'must fill', 'empty', 'not found', 'timeout', 'unrecognized',
                      'not supported', 'not allowed', 'incomplete', 'please fill']
    
    critical_count = sum(1 for kw in critical_errors if kw in state_lower)
    warning_count = sum(1 for kw in warning_errors if kw in state_lower)
    
    if critical_count >= 1:
        error_penalty = -0.25 * min(critical_count, 3)
    if warning_count >= 1:
        error_penalty -= 0.08 * min(warning_count, 4)
    
    # Form presence and context check
    form_present = any(
        kw in state_lower for kw in ['form', 'input', 'text', 'field', 'box', 'textarea', 'select', 'button']
    )
    
    task_context_present = any(
        kw in state_lower for kw in ['calendar', 'todo', 'messenger', 'maps', 'code', 'editor',
                                      'event', 'task', 'message', 'location', 'file', 'document']
    )
    
    # Form completion - count filled fields
    filled_patterns = [
        r'value="[^"]*[^"\s]',
        r'value=\'[^\']*[^\'\s]',
        r'checked',
        r'selected',
        r'type="date"[^>]*value',
        r'type="time"[^>]*value',
    ]
    
    filled_count = 0
    for pattern in filled_patterns:
        if re.search(pattern, state_lower):
            filled_count += 1
    
    # Form completion calculation - expanded ranges
    if not form_present:
        form_completion = 0.0
    elif not task_context_present:
        form_completion = 0.02
    elif filled_count == 0:
        form_completion = 0.03
    elif filled_count == 1:
        form_completion = 0.15
    elif filled_count == 2:
        form_completion = 0.28
    elif filled_count == 3:
        form_completion = 0.38
    elif filled_count >= 4:
        form_completion = 0.48
    
    # Navigation progress
    nav_keywords = ['calendar', 'todo', 'messenger', 'maps', 'code', 'editor',
                    'event', 'task', 'message', 'location', 'file', 'document',
                    'inbox', 'compose', 'settings', 'profile', 'dashboard',
                    'list', 'view', 'page', 'home', 'main', 'menu', 'app',
                    'new', 'create', 'add', 'edit', 'detail', 'open', 'active']
    
    nav_match_count = sum(1 for kw in nav_keywords if kw in state_lower)
    
    if task_context_present:
        if nav_match_count >= 10:
            navigation_progress = 0.10
        elif nav_match_count >= 7:
            navigation_progress = 0.06
        elif nav_match_count >= 5:
            navigation_progress = 0.03
        elif nav_match_count >= 3:
            navigation_progress = 0.01
        else:
            navigation_progress = 0.0
    else:
        navigation_progress = 0.0
    
    # Action efficiency - expanded ranges
    productive_actions = ['submitted', 'clicked', 'pressed', 'opened', 'selected',
                          'checked', 'toggled', 'enabled', 'removed', 'deleted',
                          'edited', 'modified', 'updated', 'changed', 'confirmed',
                          'accepted', 'saved', 'uploaded', 'downloaded', 'navigated',
                          'loaded', 'rendered', 'displayed', 'typed', 'entered',
                          'wrote', 'composed', 'drafted', 'scheduled', 'planned', 'set']
    
    action_count = sum(1 for kw in productive_actions if kw in state_lower)
    
    if action_count >= 5 and (form_completion > 0.25 or navigation_progress > 0.06):
        action_efficiency = 0.15
    elif action_count >= 3 and (form_completion > 0.15 or navigation_progress > 0.03):
        action_efficiency = 0.10
    elif action_count >= 1 and (form_completion > 0.05 or navigation_progress > 0.01):
        action_efficiency = 0.05
    else:
        action_efficiency = 0.0
    
    # Combined progress score for wandering detection
    progress_score = form_completion + navigation_progress + action_efficiency
    
    # Wandering penalty - calibrated for turn-based efficiency
    if action_count >= 2 and progress_score < 0.10:
        wandering_penalty = -0.12
    elif action_count >= 4 and progress_score < 0.15:
        wandering_penalty = -0.20
    elif action_count >= 6 and progress_score < 0.20:
        wandering_penalty = -0.30
    elif action_count >= 8 and progress_score < 0.25:
        wandering_penalty = -0.40
    elif action_count >= 10 and progress_score < 0.30:
        wandering_penalty = -0.50
    
    # Additional wandering detection
    button_count = state_lower.count('button') + state_lower.count('click') + state_lower.count('link')
    if button_count >= 8 and progress_score < 0.12:
        wandering_penalty = min(wandering_penalty - 0.08, -0.55)
    
    # Turn extraction - EXPANDED patterns for better matching
    current_turn = 0
    
    # Try multiple patterns for turn/step extraction - more comprehensive
    turn_patterns = [
        r'turn[:\s]+(\d+)',
        r'step[:\s]+(\d+)',
        r'turn\s+(\d+)',
        r'step\s+(\d+)',
        r'turn #(\d+)',
        r'step #(\d+)',
        r'(\d+)\s+turn',
        r'(\d+)\s+step',
        r'current turn[:\s]+(\d+)',
        r'current step[:\s]+(\d+)',
        r'episode turn[:\s]+(\d+)',
        r'episode step[:\s]+(\d+)',
        r'turn\s+(\d+)\s+of',
        r'step\s+(\d+)\s+of',
        r'\b(\d+)\b.*turn',
        r'\b(\d+)\b.*step',
    ]
    
    for pattern in turn_patterns:
        match = re.search(pattern, state_lower)
        if match:
            try:
                current_turn = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # Turn efficiency - AGGRESSIVE penalties for late turns with low progress
    if current_turn > 0:
        if current_turn > 40 and progress_score < 0.15:
            turn_efficiency = -0.55
        elif current_turn > 35 and progress_score < 0.20:
            turn_efficiency = -0.45
        elif current_turn > 30 and progress_score < 0.25:
            turn_efficiency = -0.35
        elif current_turn > 25 and progress_score < 0.30:
            turn_efficiency = -0.25
        elif current_turn > 20 and progress_score < 0.35:
            turn_efficiency = -0.15
        elif current_turn > 15 and progress_score < 0.40:
            turn_efficiency = -0.08
        elif current_turn > 10 and progress_score < 0.45:
            turn_efficiency = -0.04
        elif current_turn > 5 and progress_score < 0.50:
            turn_efficiency = -0.02
        elif current_turn <= 5 and progress_score > 0.30:
            turn_efficiency = 0.03
        elif current_turn <= 10 and progress_score > 0.40:
            turn_efficiency = 0.02
        elif current_turn <= 15 and progress_score > 0.35:
            turn_efficiency = 0.01
        else:
            turn_efficiency = 0.0
    else:
        turn_efficiency = 0.0
    
    # Completion proximity
    progress_indicators = 0
    if filled_count > 0:
        progress_indicators += 1
    if action_count > 0:
        progress_indicators += 1
    if nav_match_count >= 7 and task_context_present:
        progress_indicators += 1
    if any(phrase in state_lower for phrase in ['submitted', 'form submitted', 'submitted successfully', 'action completed', 'ready to submit', 'final', 'confirm']):
        progress_indicators += 1
    if form_completion >= 0.25:
        progress_indicators += 1
    if task_context_present and form_completion > 0.10:
        progress_indicators += 1
    
    if progress_indicators >= 5:
        completion_proximity = 0.28
    elif progress_indicators >= 4:
        completion_proximity = 0.20
    elif progress_indicators >= 3:
        completion_proximity = 0.12
    elif progress_indicators >= 2:
        completion_proximity = 0.06
    else:
        completion_proximity = 0.0
    
    # Calculate total value
    total = goal_success + error_penalty + form_completion + navigation_progress + action_efficiency + wandering_penalty + completion_proximity + turn_efficiency
    
    # Clamp to reasonable range [0.0, 1.0]
    total = max(0.0, min(1.0, total))
    
    return total, {
        "goal_success": goal_success,
        "error_penalty": error_penalty,
        "form_completion": form_completion,
        "navigation_progress": navigation_progress,
        "action_efficiency": action_efficiency,
        "wandering_penalty": wandering_penalty,
        "completion_proximity": completion_proximity,
        "turn_efficiency": turn_efficiency,
    }