import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) by analyzing the state transition and the 
    semantic meaning of the action taken.
    """
    next_state_lower = next_state.lower()
    
    # 1. Immediate Reward/Goal Completion Check
    # If the next state contains common success indicators for the OpenApps tasks.
    success_indicators = [
        'success', 'sent', 'added', 'saved', 'created', 
        'done', 'complete', 'confirmed', 'ok', 'message sent',
        'event added', 'task added', 'appointment created'
    ]
    for indicator in success_indicators:
        if indicator in next_state_lower:
            return 0.98

    # 2. Action Parsing
    # Extract action type, target element (bid), and supplementary info (text or key).
    act_type = None
    target_bid = None
    fill_text = None
    key_pressed = None
    
    if action.startswith('click'):
        act_type = 'click'
        m = re.search(r"click\('([^']+)'\)", action)
        if m:
            target_bid = m.group(1)
    elif action.startswith('fill'):
        act_type = 'fill'
        m = re.search(r"fill\('([^']+)',\s*'([^']+)'\)", action)
        if m:
            target_bid = m.group(1)
            fill_text = m.group(2)
    elif action.startswith('press'):
        act_type = 'press'
        m = re.search(r"press\('([^']+)',\s*'([^']+)'\)", action)
        if m:
            target_bid = m.group(1)
            key_pressed = m.group(2)
    elif action.startswith('scroll'):
        act_type = 'scroll'
    elif action.startswith('noop'):
        act_type = 'noop'
    
    if not act_type:
        return 0.0

    # 3. Evaluate State Transition
    # Determine if the action resulted in a tangible change in the accessibility tree.
    state_changed = (len(next_state) != len(state) or next_state.strip() != state.strip())

    # 4. Heuristic Scoring based on Action Type and semantic context
    
    if act_type == 'click' and target_bid:
        # Check if the clicked element's label in the state suggests it is a terminal action.
        terminal_keywords = {
            'submit', 'send', 'save', 'add', 'done', 'create', 
            'post', 'confirm', 'search', 'go', 'delete', 'apply'
        }
        bid_pattern = re.escape(target_bid)
        # Look for the bid in the state and check the text within the immediate vicinity.
        context_match = re.search(rf"{bid_pattern}.{{0,40}}", state)
        is_terminal_click = False
        if context_match:
            context = context_match.group(0).lower()
            if any(tk in context for tk in terminal_keywords):
                is_terminal_click = True
        
        if is_terminal_click:
            # Terminal actions are highly valued if they trigger a state change (e.g., navigation or confirmation).
            return 0.85 if state_changed else 0.2
        # Non-terminal clicks that change the state are indicative of progress (e.g., opening a menu).
        return 0.4 if state_changed else 0.1

    if act_type == 'fill' and target_bid:
        # If the text being entered is now found in the next_state, it's a successful interaction.
        if fill_text and fill_text.lower() in next_state_lower:
            return 0.65
        # If the fill action caused a state change (e.g., validation message or value persistence).
        return 0.3 if state_changed else 0.1

    if act_type == 'press' and target_bid:
        # Pressing Enter or Return is often used to submit forms or confirm selections.
        if key_pressed and key_pressed.lower() in ('enter', 'return'):
            return 0.55 if state_changed else 0.1
        return 0.1

    # 5. General Productivity and Baseline
    # Reward other potentially productive actions (click/fill) that lead to a change.
    if act_type in ('click', 'fill', 'press') and state_changed:
        return 0.2
    
    # Return a very low baseline for noop, scroll, or non-productive actions.
    return 0.01