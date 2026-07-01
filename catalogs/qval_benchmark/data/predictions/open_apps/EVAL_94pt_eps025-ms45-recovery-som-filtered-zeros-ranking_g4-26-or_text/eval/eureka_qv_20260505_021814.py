import re

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value Q(s, a) for OpenApps web tasks.
    Provides a dense signal using state potential, action utility, 
    transition bonuses, and detour penalties, scaled to be within [0, 1].
    """
    next_s = next_state.lower()
    curr_s = state.lower()
    act = action.lower()

    # 1. Keyword Definitions
    success_kws = ["success", "completed", "sent", "saved", "added", "confirmed", "delivered", "created", "done", "finished", "updated", "received"]
    action_kws = ["submit", "send", "save", "confirm", "post", "update", "create", "done", "search", "apply", "schedule", "add", "go"]
    task_kws = ["input", "textarea", "field", "type", "edit", "message", "text", "subject", "body", "content", "write", "button", "new", "event", "task", "todo", "contact", "name", "address", "location", "editor", "code", "calendar", "messenger", "email", "note"]
    nav_kws = ["menu", "home", "dashboard", "settings", "navigation", "back", "list", "sidebar", "profile", "account", "apps", "all-apps", "main"]
    abort_kws = ["cancel", "exit", "logout", "clear", "reset"]

    # 2. Terminal State Detection
    if any(kw in next_s for kw in success_kws):
        return 1.0, {
            "terminal_success": 1.0,
            "state_potential": 0.0,
            "action_utility": 0.0,
            "transition_bonus": 0.0,
            "detour_penalty": 0.0
        }

    # 3. State Classification
    is_action_ready = any(kw in next_s for kw in action_kws)
    is_task = any(kw in next_s for kw in task_kws)
    is_app = any(kw in next_s for kw in ["calendar", "messenger", "maps", "todo", "code", "editor"])
    is_nav = any(kw in next_s for kw in nav_kws)
    is_abort = any(kw in next_s for kw in abort_kws)
    
    was_nav = any(kw in curr_s for kw in nav_kws)
    was_task = any(kw in curr_s for kw in task_kws)

    # 4. Component: State Potential (V(s'))
    # Hierarchy of specificity: Action Ready > Task/Input State > App State > Navigation State
    if is_action_ready:
        state_potential = 0.5
    elif is_task:
        state_potential = 0.3
    elif is_app:
        state_potential = 0.2
    elif is_nav:
        state_potential = 0.1
    else:
        state_potential = 0.1

    # 5. Component: Action Utility (A(s, a))
    action_utility = 0.0
    if "fill" in act:
        # Check if the fill action contains content
        match = re.search(r"fill\s*\(\s*[^,]+,\s*['\"]([^'\"]*)['\"]", action)
        val = match.group(1).strip() if match else ""
        action_utility = 0.3 if val else -0.2
    elif "click" in act:
        if len(next_s) == len(curr_s) and next_s.strip() == curr_s.strip():
            # Penalize ineffective clicks that don't change the accessibility tree
            action_utility = -0.3
        elif is_action_ready:
            action_utility = 0.3
        elif is_task:
            action_utility = 0.1
        elif is_nav:
            # Discriminative/Neutral for early navigation to prevent pessimism
            action_utility = 0.0
        else:
            action_utility = 0.0
    elif "press" in act and "enter" in act:
        action_utility = 0.2
    elif "noop" in act or "scroll" in act:
        action_utility = -0.1

    # 6. Component: Transition Bonus (T(s, s'))
    transition_bonus = 0.0
    if was_nav and (is_app or is_task):
        transition_bonus = 0.15
    if was_task and is_action_ready:
        transition_bonus = 0.2

    # 7. Component: Detour Penalty (D(s, s'))
    detour_penalty = 0.0
    if is_abort:
        detour_penalty = -0.3
    elif was_task and is_nav:
        detour_penalty = -0.2
    
    # 8. Final Q-Value Calculation and Scaling
    raw_total = state_potential + action_utility + transition_bonus + detour_penalty
    clamped_total = max(0.0, min(1.0, raw_total))
    
    # Maintain mathematical consistency in dictionary components
    diff = clamped_total - raw_total
    state_potential += diff
    
    return clamped_total, {
        "terminal_success": 0.0,
        "state_potential": state_potential,
        "action_utility": action_utility,
        "transition_bonus": transition_bonus,
        "detour_penalty": detour_penalty
    }