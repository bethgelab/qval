def signal_function(state: str):
    import re
    
    state_lower = state.lower()
    
    # Check for task completion - return exactly 1.0 for terminal states
    success_patterns = ['event created', 'event saved', 'task completed', 'message sent', 
                        'added successfully', '✓', 'completed successfully', 'done!', 
                        'created successfully', 'saved successfully', 'successfully added',
                        'event added', 'task added', 'message delivered', 'success',
                        '✔', 'completed', 'finished', 'sent successfully']
    is_complete = any(p in state_lower for p in success_patterns)
    
    if is_complete:
        return 1.0, {"task_complete": 1.0}
    
    # Check for error states with severity levels
    error_patterns = ['error', 'failed', 'invalid', 'cannot', 'unable', '✗', '×', 
                      'failed to', 'does not exist', 'not found', 'unavailable',
                      'validation', 'required', 'missing', 'incorrect', 'wrong', 'denied']
    error_count = sum(1 for p in error_patterns if p in state_lower)
    has_error = error_count > 0
    
    # Scale error penalty based on severity - more errors = worse state
    error_penalty = -0.15 * min(error_count, 3) - 0.05 * max(error_count - 3, 0)
    
    # Detect app context with more specific patterns
    is_calendar = any(p in state_lower for p in ['calendar', 'schedule', 'event', 'agenda', 'date picker'])
    is_todo = any(p in state_lower for p in ['todo', 'task', 'checklist', 'task list'])
    is_messenger = any(p in state_lower for p in ['messenger', 'message', 'inbox', 'chat', 'conversation'])
    is_maps = any(p in state_lower for p in ['maps', 'location', 'address', 'route', 'direction'])
    is_editor = any(p in state_lower for p in ['editor', 'code', 'file', 'terminal'])
    
    # Determine primary app with priority
    app_context = None
    if is_calendar:
        app_context = 'calendar'
    elif is_todo:
        app_context = 'todo'
    elif is_messenger:
        app_context = 'messenger'
    elif is_maps:
        app_context = 'maps'
    elif is_editor:
        app_context = 'editor'
    
    # Detect step count for temporal decay
    step_count = 0
    step_match = re.search(r'(?:step|turn)[:\s]*(\d+)', state_lower)
    if step_match:
        try:
            step_count = int(step_match.group(1))
            if step_count < 0 or step_count > 45:
                step_count = 0
        except:
            pass
    
    # Calculate temporal decay factor
    max_steps = 45
    remaining_steps = max(max_steps - step_count, 0)
    temporal_decay = 0.2 + 0.8 * (remaining_steps / max_steps) ** 1.1
    
    # Check for empty/initial state
    empty_patterns = ['empty', 'no events', 'no tasks', 'nothing', 'blank', 'no items',
                      'no messages', 'no conversations', 'no results', 'no data']
    is_empty = any(p in state_lower for p in empty_patterns)
    
    # Detect form state
    form_patterns = ['create', 'add', 'new', 'form', 'event name', 'title', 'description', 
                     'date', 'time', 'from', 'to', 'recipient', 'start', 'end', 'subject',
                     'compose', 'write', 'enter', 'fill', 'input', 'textbox', 'textarea']
    in_form = any(p in state_lower for p in form_patterns)
    
    # View type detection with better app-specific patterns
    view_type = 'unknown'
    if 'inbox' in state_lower or 'messages' in state_lower or 'conversations' in state_lower:
        view_type = 'inbox'
    elif 'compose' in state_lower or 'new message' in state_lower or 'write' in state_lower:
        view_type = 'compose'
    elif 'conversation' in state_lower or 'chat' in state_lower or 'thread' in state_lower:
        view_type = 'conversation'
    elif 'event' in state_lower and 'create' in state_lower:
        view_type = 'create_event'
    elif 'calendar' in state_lower and 'schedule' in state_lower:
        view_type = 'calendar_view'
    elif 'todo' in state_lower or 'tasks' in state_lower:
        view_type = 'task_list'
    elif 'code' in state_lower or 'editor' in state_lower:
        view_type = 'code_editor'
    elif 'maps' in state_lower or 'location' in state_lower:
        view_type = 'maps_view'
    
    # Count form input elements
    input_count = sum(1 for p in ['input', 'textbox', 'textarea', 'field', 'edit', 
                                   'editable', 'placeholder'] if p in state_lower)
    
    # Count filled form fields using regex
    value_matches = re.findall(r'value=["\']([^"\']+)["\']', state)
    filled_field_count = len([m for m in value_matches if m.strip() and len(m) > 2])
    
    # Count specific form elements for each app type
    calendar_form_elements = sum(1 for p in ['event name', 'event title', 'date', 'time', 
                                              'start time', 'end time', 'description', 'location',
                                              'calendar', 'schedule'] if p in state_lower)
    
    todo_form_elements = sum(1 for p in ['task', 'title', 'description', 'due date', 'priority',
                                          'todo', 'checklist', 'assign'] if p in state_lower)
    
    messenger_form_elements = sum(1 for p in ['message', 'recipient', 'to:', 'from:', 'subject', 
                                               'body', 'compose', 'chat', 'conversation'] if p in state_lower)
    
    # Calculate form completion level
    if in_form and calendar_form_elements > 0:
        completion_level = min(filled_field_count / max(calendar_form_elements, 1), 1.0)
    elif in_form and todo_form_elements > 0:
        completion_level = min(filled_field_count / max(todo_form_elements, 1), 1.0)
    elif in_form and messenger_form_elements > 0:
        completion_level = min(filled_field_count / max(messenger_form_elements, 1), 1.0)
    elif in_form and input_count > 0:
        completion_level = min(filled_field_count / max(input_count, 1), 1.0)
    else:
        completion_level = 0.0
    
    # Enhanced proximity bonus based on submit-ready indicators
    proximity_patterns = ['save', 'submit', 'create', 'add', 'send', 'confirm', 
                          'finish', 'done', 'complete', 'next', 'continue', '✓']
    proximity_count = sum(1 for p in proximity_patterns if p in state_lower)
    
    # Scale proximity bonus based on form state and proximity indicators
    proximity_bonus = 0.0
    if in_form and proximity_count > 0 and completion_level > 0.3:
        proximity_bonus = 0.08 + 0.04 * min(proximity_count, 3)
    elif in_form and proximity_count > 0:
        proximity_bonus = 0.04 + 0.02 * min(proximity_count, 3)
    elif proximity_count > 0:
        proximity_bonus = 0.02 + 0.01 * min(proximity_count, 3)
    
    # View-specific bonus with better app differentiation
    view_bonus = 0.0
    if app_context == 'messenger':
        if view_type == 'compose':
            view_bonus = 0.12
        elif view_type == 'conversation':
            view_bonus = 0.08
        elif view_type == 'inbox':
            view_bonus = 0.04
    elif app_context == 'calendar':
        if view_type == 'create_event':
            view_bonus = 0.14
        elif view_type == 'calendar_view':
            view_bonus = 0.06
    elif app_context == 'todo':
        if view_type == 'task_list':
            view_bonus = 0.05
    elif app_context == 'maps':
        if view_type == 'maps_view':
            view_bonus = 0.03
    
    # App context bonus - more differentiated values
    page_match = 0.0
    if app_context == 'calendar':
        page_match = 0.20
    elif app_context == 'messenger':
        page_match = 0.16
    elif app_context == 'todo':
        page_match = 0.12
    elif app_context == 'editor':
        page_match = 0.06
    elif app_context == 'maps':
        page_match = 0.02
    else:
        page_match = -0.05
    
    # Base value
    base_value = 0.03
    
    # Navigation progress - more specific to correct app
    nav_indicators = ['calendar', 'events', 'schedule', 'todo', 'tasks', 'messages', 'inbox', 'chat']
    has_nav_indicator = any(p in state_lower for p in nav_indicators)
    nav_progress = 0.0
    if app_context and has_nav_indicator:
        nav_progress = 0.06
    elif has_nav_indicator and not app_context:
        nav_progress = 0.03
    elif app_context and not has_nav_indicator:
        nav_progress = -0.02
    
    # Content presence bonus
    if not is_empty:
        base_value += 0.02
    
    # Form progress
    form_progress = 0.12 if in_form else 0.02
    
    # Scale progress reward with temporal decay and completion
    progress_reward = (form_progress + completion_level * 0.35 + filled_field_count * 0.03) * temporal_decay
    
    # Empty penalty
    empty_penalty = -0.05 if is_empty else 0.0
    
    # Calculate total
    total = base_value + progress_reward + page_match + empty_penalty + error_penalty + proximity_bonus + view_bonus + nav_progress
    
    # Cap incomplete states at 0.95, completed states already return 1.0
    total = max(0.0, min(0.95, total))
    
    return total, {
        "base_value": base_value,
        "progress_reward": progress_reward,
        "page_match": page_match,
        "empty_penalty": empty_penalty,
        "error_penalty": error_penalty,
        "proximity_bonus": proximity_bonus,
        "view_bonus": view_bonus,
        "nav_progress": nav_progress,
    }