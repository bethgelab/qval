def signal_function(state: str, action: str, next_state: str):
    import re
    
    # Parse action type from BrowserGym action string
    action_lower = action.lower()
    action_match = re.match(r'(\w+)\(', action_lower)
    action_type = action_match.group(1) if action_match else 'unknown'
    
    # Extract bid from action for precise tracking
    bid_pattern = r"['\"](\d+)['\"]"
    action_bid = re.search(bid_pattern, action)
    action_bid_num = int(action_bid.group(1)) if action_bid else None
    
    # Count interactive elements (bids) in states using multiple patterns
    bid_patterns = [
        r'\bbid\d+\b',
        r'\[bid\s*\d+\]',
        r'\bid\s+\d+\b',
    ]
    
    state_bids = 0
    for pattern in bid_patterns:
        state_bids += len(re.findall(pattern, state, re.IGNORECASE))
    
    next_state_bids = 0
    for pattern in bid_patterns:
        next_state_bids += len(re.findall(pattern, next_state, re.IGNORECASE))
    
    # Calculate bid changes
    bid_change = next_state_bids - state_bids
    new_bids = max(0, next_state_bids - state_bids)
    removed_bids = max(0, state_bids - next_state_bids)
    
    # Detect regression - navigation away from target app
    regression_penalty = 0.0
    
    # Check for exit/navigation patterns with more robust matching
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    
    # Exit patterns - broader matching for detecting navigation away from task
    exit_patterns = ['home', 'main menu', 'exit', 'close', 'dashboard', 
                     'index', 'landing', 'logout', 'sign out', 'back to home',
                     'main page', 'home page', 'welcome', 'start page', 'menu']
    is_exit_pattern = any(pattern in next_state_lower for pattern in exit_patterns)
    
    # Check for significant bid reduction indicating page change
    significant_bid_loss = removed_bids >= 4
    
    # Check if we're losing the current app context
    app_keywords = ['todo', 'calendar', 'messenger', 'maps', 'editor', 'code',
                    'task', 'event', 'message', 'note', 'document']
    had_app_context = any(kw in state_lower for kw in app_keywords)
    lost_app_context = not any(kw in next_state_lower for kw in app_keywords)
    
    # Enhanced regression detection with more granular thresholds
    if is_exit_pattern and had_app_context:
        regression_penalty = -0.30 - (removed_bids * 0.028)
    elif significant_bid_loss and had_app_context and lost_app_context:
        regression_penalty = -0.24 - (removed_bids * 0.020)
    elif removed_bids >= 3 and action_type == 'click' and had_app_context:
        regression_penalty = -0.16
    elif removed_bids >= 2 and action_type in ['click', 'press'] and had_app_context:
        regression_penalty = -0.11
    elif removed_bids >= 1 and action_type in ['click', 'press'] and had_app_context and not is_exit_pattern:
        regression_penalty = -0.06
    
    # Detect form progress - high-value task steps
    form_progress_bonus = 0.0
    form_keywords = ['form', 'input', 'filled', 'entered', 'typed', 'text', 
                     'field', 'textarea', 'value', 'select', 'option', 'dropdown']
    
    if action_type == 'fill':
        if any(kw in next_state_lower for kw in form_keywords):
            form_progress_bonus = 0.28
        else:
            form_progress_bonus = 0.22
    elif action_type == 'press':
        if any(kw in next_state_lower for kw in form_keywords):
            form_progress_bonus = 0.19
        else:
            form_progress_bonus = 0.14
    elif action_type == 'click':
        if any(kw in next_state_lower for kw in form_keywords):
            form_progress_bonus = 0.13
    
    # Detect navigation (moderate value for exploration)
    navigation_bonus = 0.0
    nav_progress_keywords = ['navigated', 'redirected', 'moved', 'page', 'view', 
                             'loaded', 'displayed', 'opened', 'shown']
    
    if action_type in ['click', 'press'] and any(kw in next_state_lower for kw in nav_progress_keywords):
        if not is_exit_pattern and removed_bids < 4:
            navigation_bonus = 0.10
    
    # Combine form and navigation bonuses
    progress_bonus = form_progress_bonus + navigation_bonus
    progress_bonus = min(0.38, progress_bonus)
    
    # Bid change signal - new elements indicate progress
    bid_signal = 0.0
    if new_bids > 0:
        bid_signal = min(0.10, new_bids * 0.024)
    elif removed_bids > 5:
        bid_signal = -0.08
    
    # Content change signal - substantial text changes
    state_len = len(state)
    next_len = len(next_state)
    content_diff = next_len - state_len
    content_signal = 0.0
    
    if abs(content_diff) > 120:
        content_signal = min(0.10, abs(content_diff) / 4000)
        if content_diff < 0 and removed_bids == 0:
            content_signal = -0.045
    
    # Milestone keywords - significant progress indicators
    milestone_keywords = {
        'success': 0.32, 'complete': 0.32, 'done': 0.30,
        'saved': 0.28, 'created': 0.28, 'added': 0.26,
        'sent': 0.32, 'submitted': 0.32, 'confirmed': 0.32,
        'verified': 0.30, 'finalized': 0.30, 'finished': 0.30,
        'form': 0.14, 'dialog': 0.14, 'modal': 0.14, 'popup': 0.12,
        'opened': 0.12, 'loaded': 0.12, 'displayed': 0.10, 'shown': 0.10,
        'error': -0.15, 'failed': -0.17, 'invalid': -0.15
    }
    
    milestone_bonus = 0.0
    for keyword, weight in milestone_keywords.items():
        if keyword in next_state_lower:
            if weight > 0:
                milestone_bonus = max(milestone_bonus, weight)
            else:
                milestone_bonus = min(milestone_bonus, weight)
    
    # App context bonus
    task_keywords = {
        'todo': 0.05, 'calendar': 0.05, 'messenger': 0.05,
        'maps': 0.05, 'code': 0.05, 'editor': 0.05,
        'event': 0.04, 'task': 0.04, 'message': 0.04,
        'app': 0.03, 'application': 0.03, 'note': 0.03
    }
    context_bonus = sum(weight for kw, weight in task_keywords.items() if kw in state_lower)
    context_bonus = min(0.16, context_bonus)
    
    # Action-specific base reward
    if action_type == 'fill':
        action_reward = 0.39
    elif action_type == 'click':
        action_reward = 0.30
    elif action_type == 'press':
        action_reward = 0.27
    elif action_type == 'scroll':
        action_reward = 0.08
    elif action_type == 'noop':
        action_reward = -0.20
    else:
        action_reward = 0.0
    
    # Efficiency penalty - penalize unproductive actions (calibrated for long-horizon)
    efficiency_penalty = 0.0
    has_meaningful_change = (abs(bid_change) > 0 or abs(content_diff) > 80 or milestone_bonus > 0.18 or form_progress_bonus > 0.15)
    
    if action_type == 'noop':
        efficiency_penalty = -0.20
    elif action_type in ['click', 'fill', 'press']:
        if not has_meaningful_change and milestone_bonus == 0 and form_progress_bonus == 0:
            efficiency_penalty = -0.08
    elif action_type == 'scroll':
        if not has_meaningful_change and abs(content_diff) < 40:
            efficiency_penalty = -0.05
        else:
            efficiency_penalty = 0.0
    
    # Momentum bonus - reward consistent productive patterns
    momentum_bonus = 0.0
    positive_signals = sum([
        progress_bonus > 0.11,
        new_bids > 0,
        abs(content_diff) > 120,
        milestone_bonus > 0.18
    ])
    if positive_signals >= 2:
        momentum_bonus = 0.055 * positive_signals
    
    # Combine signals with milestone-aware weighting
    if milestone_bonus > 0.28:
        progress_reward = (progress_bonus * 1.55 + bid_signal * 1.3 + content_signal * 1.0)
    elif milestone_bonus > 0.18:
        progress_reward = (progress_bonus * 1.3 + bid_signal * 1.1 + content_signal * 0.8)
    elif milestone_bonus > 0.10:
        progress_reward = (progress_bonus * 1.1 + bid_signal * 0.9 + content_signal * 0.6)
    else:
        progress_reward = (progress_bonus * 0.9 + bid_signal * 0.7 + content_signal * 0.4)
    
    # Scale progress reward by action type effectiveness
    if action_type in ['fill', 'click', 'press']:
        progress_reward = min(0.48, max(-0.11, progress_reward * 1.3))
    elif action_type == 'scroll':
        progress_reward = min(0.19, max(-0.08, progress_reward * 0.9))
    elif action_type == 'noop':
        progress_reward = min(0.06, max(-0.15, progress_reward * 0.3 - 0.11))
    else:
        progress_reward = 0.0
    
    # Calculate total Q-value estimate
    total = action_reward + progress_reward + milestone_bonus + context_bonus + efficiency_penalty + momentum_bonus + regression_penalty
    
    # Cap at 0.82 for non-terminal states to avoid over-optimism and maintain gradient signal
    task_completion_keywords = ['success', 'complete', 'done', 'saved', 'created', 'added', 'sent', 'submitted', 'confirmed']
    is_likely_complete = any(kw in next_state_lower for kw in task_completion_keywords)
    
    if is_likely_complete:
        total = min(0.98, max(-0.5, total))
    else:
        total = min(0.82, max(-0.5, total))
    
    return total, {
        "action_reward": action_reward,
        "progress_reward": progress_reward,
        "milestone_bonus": milestone_bonus,
        "context_bonus": context_bonus,
        "efficiency_penalty": efficiency_penalty,
        "momentum_bonus": momentum_bonus,
        "regression_penalty": regression_penalty,
    }