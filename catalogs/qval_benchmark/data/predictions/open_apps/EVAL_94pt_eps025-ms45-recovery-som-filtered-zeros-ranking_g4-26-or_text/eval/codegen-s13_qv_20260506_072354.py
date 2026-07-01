import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a given state-action-next_state transition.
    High Q-values are assigned to actions that lead to task completion or
    demonstrate progress towards the goal.
    """
    # 1. Immediate Reward Check: If the next state indicates the task is complete.
    # In OpenApps, success is often indicated by confirmation text.
    success_indicators = [
        'success', 'sent', 'added', 'created', 'saved', 'updated', 
        'completed', 'confirmed', 'done', 'message sent', 'event created'
    ]
    if any(indicator in next_state.lower() for indicator in success_indicators):
        return 1.0

    # 2. Action Parsing
    # Standard BrowserGym action primitives: click('bid'), fill('bid', 'text'), press('bid', 'key'), scroll(x, y), noop(ms)
    is_noop = 'noop' in action
    if is_noop:
        return 0.0

    # Regex to extract command and parameters
    click_match = re.search(r"click\(['\"](\d+)['\"]\)", action)
    fill_match = re.search(r"fill\(['\"](\d+)['\"]\s*,\s*['\"]([^'\"]*)['\"]\)", action)
    press_match = re.search(r"press\(['\"](\d+)['\"]\s*,\s*['\"]([^'\"]*)['\"]\)", action)

    # Base value for a valid, non-noop action
    q_val = 0.1

    # 3. Analyze the Action's potential for progress
    if click_match:
        bid = click_match.group(1)
        # Try to find the semantic meaning of the clicked element in the accessibility tree
        # Elements in state usually look like: [bid: 5, role: button, name: "Submit"]
        element_pattern = rf"bid:\s*'?{bid}'?.*? (?:name|text):\s*['\"]([^'\"]*)['\"]"
        element_match = re.search(element_pattern, state, re.DOTALL)
        
        if element_match:
            element_text = element_match.group(1).lower()
            # Keywords indicating actions that typically move a user closer to a goal
            productive_kws = ['submit', 'send', 'save', 'add', 'create', 'confirm', 'ok', 'search', 'go', 'apply', 'delete', 'edit']
            if any(kw in element_text for kw in productive_kws):
                q_val += 0.6  # High value for clicking a "primary" action button
            else:
                q_val += 0.2  # Generic click might be navigation or selection
        else:
            q_val += 0.2  # Element text not found, but a click is still an intentional movement

    elif fill_match:
        # For 'fill', the content being entered is a strong signal of progress
        content = fill_match.group(2)
        if content.strip():
            q_val += 0.4  # Providing information is a key step in most tasks
        else:
            q_val += 0.0  # Filling empty strings is usually a low-value/error action

    elif press_match:
        key = press_match.group(2).lower()
        if key in ['enter', 'return']:
            q_val += 0.3  # Enter key is often used to submit forms
        else:
            q_val += 0.1

    # 4. State Transition Analysis
    # A high-quality action should result in a visible change in the environment (state).
    if next_state.strip() == state.strip():
        # If the action resulted in no state change, it's likely a wasted step.
        q_val *= 0.5
    elif q_val > 0.1:
        # If the action was productive AND the state changed, reinforce the value.
        q_val += 0.2

    # Ensure the return value is within the logical bounds of [0.0, 1.0]
    return min(max(q_val, 0.0), 1.0)