def signal_function(state: str, action: str, next_state: str):
    import re
    
    # Goal-related keywords for web apps
    goal_keywords = [
        'submit', 'save', 'create', 'add', 'send', 'post',
        'complete', 'done', 'success', 'event', 'message',
        'todo', 'calendar', 'contact', 'profile', 'edit',
        'confirmation', 'saved', 'created', 'added', 'sent',
        'task added', 'event created', 'message sent', 'item saved'
    ]
    
    # Progress indicators - things that suggest we're getting closer
    progress_indicators = [
        'selected', 'filled', 'clicked', 'completed',
        'added', 'created', 'sent', 'saved', 'updated',
        'form', 'field', 'input', 'button', 'menu',
        'dialog', 'modal', 'notification', 'toast',
        'title', 'date', 'time', 'subject', 'body'
    ]
    
    # Form field indicators for granular progress tracking
    field_indicators = [
        'input', 'text', 'textarea', 'select', 'checkbox',
        'radio', 'date', 'time', 'email', 'phone', 'name',
        'label', 'placeholder', 'required', 'value', 'field'
    ]
    
    # Error/stuck indicators
    error_indicators = [
        'error', 'invalid', 'failed', 'not found', 'blocked',
        'disabled', 'unavailable', 'timeout', 'loading',
        'error:', 'failed to', 'cannot', 'unable', 'please'
    ]
    
    # Navigation/reset indicators that should be penalized
    navigation_keywords = [
        'home', 'back', 'logout', 'sign out', 'reset',
        'cancel', 'close', 'exit', 'discard', 'clear'
    ]
    
    # Near-success indicators - completion confirmation patterns
    near_success_indicators = [
        'success', 'completed', 'saved', 'done', 'confirmation',
        'confirmation dialog', 'success message', 'task completed',
        'item created', 'event added', 'message sent', 'todo added',
        'your changes have been saved', 'successfully', 'finished',
        'task complete', 'event created', 'message sent successfully',
        'added to', 'saved successfully', 'created successfully'
    ]
    
    # Stage-based indicators for task phase detection
    early_stage_indicators = [
        'start', 'begin', 'new', 'create', 'add', 'open',
        'home', 'main', 'dashboard', 'welcome', 'initial'
    ]
    
    mid_stage_indicators = [
        'form', 'input', 'fill', 'enter', 'type', 'select',
        'choose', 'confirm', 'submit', 'save', 'update'
    ]
    
    late_stage_indicators = [
        'success', 'completed', 'done', 'finished', 'saved',
        'confirmation', 'confirmation dialog', 'task complete',
        'event created', 'message sent', 'item added'
    ]
    
    # State analysis
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Count keyword matches
    state_goal_matches = sum(1 for kw in goal_keywords if kw in state_lower)
    next_state_goal_matches = sum(1 for kw in goal_keywords if kw in next_state_lower)
    
    # Count progress indicators
    state_progress = sum(1 for kw in progress_indicators if kw in state_lower)
    next_state_progress = sum(1 for kw in progress_indicators if kw in next_state_lower)
    
    # Count field indicators for granular form progress
    state_fields = sum(1 for kw in field_indicators if kw in state_lower)
    next_state_fields = sum(1 for kw in field_indicators if kw in next_state_lower)
    
    # Count error indicators
    state_errors = sum(1 for kw in error_indicators if kw in state_lower)
    next_state_errors = sum(1 for kw in error_indicators if kw in next_state_lower)
    
    # Count near-success indicators
    state_near_success = sum(1 for kw in near_success_indicators if kw in state_lower)
    next_state_near_success = sum(1 for kw in near_success_indicators if kw in next_state_lower)
    
    # Count stage indicators
    state_early = sum(1 for kw in early_stage_indicators if kw in state_lower)
    state_mid = sum(1 for kw in mid_stage_indicators if kw in state_lower)
    state_late = sum(1 for kw in late_stage_indicators if kw in state_lower)
    
    next_early = sum(1 for kw in early_stage_indicators if kw in next_state_lower)
    next_mid = sum(1 for kw in mid_stage_indicators if kw in next_state_lower)
    next_late = sum(1 for kw in late_stage_indicators if kw in next_state_lower)
    
    # Check for navigation/reset actions
    nav_matches = sum(1 for kw in navigation_keywords if kw in next_state_lower)
    
    # Detect specific action effects
    detected_action = None
    if 'fill' in action_lower:
        detected_action = 'form_filled'
    elif 'click' in action_lower:
        detected_action = 'element_clicked'
    elif 'press' in action_lower:
        detected_action = 'key_pressed'
    elif 'scroll' in action_lower:
        detected_action = 'page_scrolled'
    elif 'noop' in action_lower:
        detected_action = 'no_action'
    
    # Calculate progress reward based on stage transitions
    progress_reward = 0.0
    
    # Stage progression bonus - refined thresholds
    stage_progress = 0.0
    if next_late > state_late:
        stage_progress = 0.38
    elif next_late >= 1 and next_mid > state_mid:
        stage_progress = 0.28
    elif next_mid > state_mid:
        stage_progress = 0.20
    elif next_mid >= 1 and next_early < state_early:
        stage_progress = 0.14
    
    # Goal keyword progression with stage awareness
    goal_diff = next_state_goal_matches - state_goal_matches
    if goal_diff >= 2:
        progress_reward = 0.48 + min(0.12, goal_diff * 0.05)
    elif goal_diff == 1:
        progress_reward = 0.32 + min(0.18, (next_state_fields - state_fields) * 0.07)
    elif goal_diff == 0 and next_late > 0:
        progress_reward = 0.28 + min(0.08, next_late * 0.03)
    
    # Progress indicators with field completion bonus
    progress_diff = next_state_progress - state_progress
    field_diff = next_state_fields - state_fields
    
    if progress_diff >= 2 or (progress_diff == 1 and field_diff >= 1):
        progress_reward = max(progress_reward, 0.38 + min(0.20, progress_diff * 0.07))
    elif progress_diff >= 1:
        progress_reward = max(progress_reward, 0.28 + min(0.16, progress_diff * 0.06))
    
    # Weak signal: goal keywords present in next state without progression
    elif next_state_goal_matches > 0:
        progress_reward = max(progress_reward, 0.22)
    
    # Near-success bonus - calibrated to catch completion early
    near_success_bonus = 0.0
    if next_state_near_success >= 5:
        near_success_bonus = 0.30
    elif next_state_near_success >= 4:
        near_success_bonus = 0.24
    elif next_state_near_success >= 3:
        near_success_bonus = 0.18
    elif next_state_near_success >= 2:
        near_success_bonus = 0.12
    elif next_state_goal_matches >= 3 and next_state_progress > state_progress:
        near_success_bonus = 0.16
    elif next_state_goal_matches >= 2 and next_state_fields > state_fields:
        near_success_bonus = 0.11
    elif next_state_goal_matches >= 1 and next_state_progress > state_progress:
        near_success_bonus = 0.07
    
    # Action bonus based on action type - more differentiated
    action_bonus = 0.0
    if detected_action == 'form_filled':
        action_bonus = 0.22
    elif detected_action == 'element_clicked':
        action_bonus = 0.16
    elif detected_action == 'key_pressed':
        action_bonus = 0.14
    elif detected_action == 'page_scrolled':
        action_bonus = 0.09
    elif detected_action == 'no_action':
        action_bonus = -0.14
    
    # Safety penalty for errors - calibrated sensitivity
    safety_penalty = 0.0
    if next_state_errors > state_errors + 1:
        safety_penalty = -0.42
    elif next_state_errors > state_errors:
        safety_penalty = -0.28
    elif next_state_errors > 0:
        safety_penalty = -0.18
    
    # Goal bonus for states with goal keywords - reduced to prevent stacking
    goal_bonus = 0.0
    if next_state_goal_matches >= 5:
        goal_bonus = 0.22
    elif next_state_goal_matches >= 4:
        goal_bonus = 0.18
    elif next_state_goal_matches >= 3:
        goal_bonus = 0.14
    elif next_state_goal_matches >= 2:
        goal_bonus = 0.10
    elif next_state_goal_matches >= 1 and next_state_progress > state_progress:
        goal_bonus = 0.06
    
    # Efficiency bonus based on progress rate - calibrated for step limit
    efficiency_bonus = 0.0
    if next_state_goal_matches >= 4:
        efficiency_bonus = 0.16
    elif next_state_goal_matches >= 3:
        efficiency_bonus = 0.12
    elif next_state_goal_matches >= 2:
        efficiency_bonus = 0.09
    elif next_state_goal_matches >= 1 and next_state_progress > state_progress:
        efficiency_bonus = 0.06
    elif next_state_progress > state_progress and next_state_fields > state_fields:
        efficiency_bonus = 0.05
    elif detected_action == 'no_action':
        efficiency_bonus = -0.10
    elif nav_matches > 0:
        efficiency_bonus = -0.08
    
    # Navigation penalty - calibrated for exploration tolerance
    nav_penalty = 0.0
    if detected_action == 'page_scrolled':
        if progress_diff <= 0 and next_state_goal_matches == state_goal_matches:
            nav_penalty = -0.20
        elif progress_diff == 0 and next_state_fields == state_fields:
            nav_penalty = -0.14
    elif detected_action == 'no_action':
        if next_state_goal_matches == state_goal_matches and next_state_progress == state_progress:
            nav_penalty = -0.22
        elif next_state_goal_matches == state_goal_matches:
            nav_penalty = -0.16
    elif nav_matches > 0 and next_state_goal_matches == 0:
        nav_penalty = -0.20
    elif nav_matches > 0 and next_state_goal_matches == state_goal_matches:
        nav_penalty = -0.12
    
    # Diminishing returns - more active based on action repetition and redundancy
    high_component_count = sum([
        1 if progress_reward > 0.4 else 0,
        1 if near_success_bonus > 0.18 else 0,
        1 if goal_bonus > 0.14 else 0,
        1 if efficiency_bonus > 0.12 else 0
    ])
    
    # Check for action repetition (noop or same action type)
    action_repetition_penalty = 0.0
    if detected_action == 'no_action':
        action_repetition_penalty = -0.10
    elif 'noop' in action_lower or 'wait' in action_lower:
        action_repetition_penalty = -0.07
    
    # Check for redundant progress (same indicators without actual advancement)
    redundancy_penalty = 0.0
    if next_state_goal_matches == state_goal_matches and next_state_progress == state_progress:
        if next_state_fields == state_fields and detected_action != 'noop':
            redundancy_penalty = -0.08
    
    diminishing_return = 0.0
    if high_component_count >= 4:
        diminishing_return = -0.18
    elif high_component_count >= 3:
        diminishing_return = -0.12
    elif high_component_count >= 2:
        diminishing_return = -0.07
    
    # Combine diminishing returns
    diminishing_return += action_repetition_penalty + redundancy_penalty
    diminishing_return = max(-0.25, diminishing_return)
    
    # Calculate total Q-value
    total = progress_reward + near_success_bonus + action_bonus + safety_penalty + goal_bonus + efficiency_bonus + nav_penalty + diminishing_return
    
    # Clamp to reasonable range - expanded for better signal in sparse settings
    total = max(0.0, min(0.95, total))
    
    return total, {
        "progress_reward": progress_reward,
        "near_success_bonus": near_success_bonus,
        "action_bonus": action_bonus,
        "safety_penalty": safety_penalty,
        "goal_bonus": goal_bonus,
        "efficiency_bonus": efficiency_bonus,
        "nav_penalty": nav_penalty,
        "diminishing_return": diminishing_return,
    }