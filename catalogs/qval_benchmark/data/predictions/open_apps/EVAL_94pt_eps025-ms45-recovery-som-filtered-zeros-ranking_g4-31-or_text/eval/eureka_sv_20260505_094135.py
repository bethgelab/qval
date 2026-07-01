import re

def signal_function(state: str):
    """
    Estimates the state-value V(s) for an agent interacting with OpenApps.
    The value is calculated based on target application identification, 
    navigation progress, input completion, and success confirmation.
    """
    state_lower = state.lower()
    
    # 1. Target Application Identification (from the goal description)
    app_keywords = {
        'todo': ['todo', 'task', 'list', 'checklist', 'item'],
        'calendar': ['calendar', 'event', 'schedule', 'appointment', 'date'],
        'messenger': ['messenger', 'message', 'chat', 'send', 'conversation', 'dm'],
        'maps': ['maps', 'map', 'location', 'route', 'direction', 'navigation', 'search for'],
        'code editor': ['code', 'editor', 'file', 'script', 'terminal', 'programming', 'javascript', 'python']
    }
    
    # Goal is typically in the first portion of the observation
    goal_part = state_lower[:2000]
    target_app = None
    for app, keywords in app_keywords.items():
        if any(k in goal_part for k in keywords):
            target_app = app
            break
            
    # 2. Current Context Detection
    hub_indicators = ['launchpad', 'welcome to openapps', 'main menu', 'all apps', 'dashboard']
    is_hub = any(h in state_lower for h in hub_indicators)
    
    current_app = None
    if not is_hub:
        # Prioritize matching against keywords to determine which app the agent is currently in
        for app, keywords in app_keywords.items():
            if any(k in state_lower for k in keywords):
                current_app = app
                break
    
    is_correct_app = (target_app is not None and current_app == target_app)
    
    # App Status Score: Hub > Wrong App
    # This ensures the agent prefers the Hub (gateway) over being stuck in the wrong application.
    if is_correct_app:
        app_status_score = 0.20
    elif is_hub:
        app_status_score = 0.10
    elif current_app is not None:
        app_status_score = -0.10
    else:
        app_status_score = 0.0
    
    # 3. Depth and Intent Progression
    # These markers indicate the agent is on the right page to perform the action.
    depth_score = 0.0
    if is_correct_app:
        if target_app == 'messenger' and any(k in state_lower for k in ['chatting with', 'conversation', 'message to', 'recipient', 'compose']):
            depth_score = 0.30
        elif target_app == 'maps' and any(k in state_lower for k in ['directions', 'route to', 'navigation', 'search results', 'nearby']):
            depth_score = 0.30
        elif target_app == 'todo' and any(k in state_lower for k in ['task details', 'edit task', 'due date', 'assignee', 'add task']):
            depth_score = 0.25
        elif target_app == 'calendar' and any(k in state_lower for k in ['event details', 'time', 'location', 'reminder', 'add event']):
            depth_score = 0.25
        elif target_app == 'code editor' and any(k in state_lower for k in ['editing', 'save file', 'terminal', 'line', 'new file']):
            depth_score = 0.25
        elif any(k in state_lower for k in ['new', 'create', 'add', 'compose', 'edit', 'search', 'plus']):
            depth_score = 0.15

    # 4. Input Filling Progress
    filled_patterns = [
        r'value=["\']([^"\']+)["\']',
        r'value\s*[:=]\s*["\']([^"\']+)["\']',
        r'value\s*[:=]\s*([^"\'\s,\]\n]+)',
        r'aria-valuenow=["\']([^"\']+)["\']'
    ]
    
    all_filled = []
    for pattern in filled_patterns:
        all_filled.extend(re.findall(pattern, state_lower))
    
    placeholders = {
        '', 'none', 'null', 'placeholder', 'search...', 'enter text', 
        'type here', 'select', 'choose', 'undefined', 'nan'
    }
    num_filled = len([v for v in all_filled if v.strip() and v.strip().lower() not in placeholders])
    
    # Filling score scales with number of fields, capped at 0.3
    filling_score = min(0.3, num_filled * 0.1)
    
    # 5. Readiness to Submit
    completion_keywords = [
        'submit', 'save', 'confirm', 'finish', 'ok', 'send', 
        'search', 'create', 'add', 'post', 'update', 'save changes', 'done'
    ]
    has_completion_btn = any(k in state_lower for k in completion_keywords)
    
    submit_ready_score = 0.0
    if has_completion_btn and is_correct_app:
        if num_filled >= 2:
            submit_ready_score = 0.30
        elif num_filled >= 1:
            submit_ready_score = 0.20
        else:
            submit_ready_score = 0.05
            
    # 6. Confirmation/Success Detection
    success_words = ['successfully', 'confirmed', 'created', 'sent', 'added', 'saved', 'updated']
    has_success_word = any(sw in state_lower for sw in success_words)
    has_completion_phrase = any(phrase in state_lower for phrase in [
        'has been created', 'has been sent', 'has been added', 'successfully saved',
        'task created', 'event created', 'message sent', 'changes saved'
    ])
    
    confirmation_score = 0.0
    if is_correct_app:
        if has_completion_phrase:
            confirmation_score = 0.70
        elif has_success_word:
            confirmation_score = 0.40
    elif has_completion_phrase:
        confirmation_score = 0.20
    
    # 7. Error Detection
    error_keywords = [
        'error', 'invalid', 'required', 'failed', 
        'wrong', 'incorrect', 'missing', 'unable', 
        'please fill', 'must be', 'not allowed', 'please enter'
    ]
    has_error = any(k in state_lower for k in error_keywords)
    error_penalty = -0.2 if has_error else 0.0
    if has_error and num_filled >= 1:
        error_penalty = -0.1 # Agent is in the process of fixing the error
            
    # 8. Stagnation Penalty
    stagnation_penalty = 0.0
    if is_correct_app and confirmation_score == 0:
        # Penalty if the agent is in the app but hasn't moved to a depth/filling state
        if depth_score == 0 and filling_score == 0:
            stagnation_penalty = -0.1
        elif depth_score > 0 and filling_score == 0:
            stagnation_penalty = -0.05
    
    # 9. Aggregation
    components = {
        "app_status_score": app_status_score,
        "depth_score": depth_score,
        "filling_score": filling_score,
        "submit_ready_score": submit_ready_score,
        "confirmation_score": confirmation_score,
        "error_penalty": error_penalty,
        "stagnation_penalty": stagnation_penalty,
    }
    
    raw_sum = sum(components.values())
    total = max(0.0, min(0.95, raw_sum))
    
    # To ensure the total is exactly the sum of components for search feedback,
    # we add any clamping difference as an adjustment.
    adjustment = total - raw_sum
    components["clamp_adjustment"] = adjustment
    
    return total, components