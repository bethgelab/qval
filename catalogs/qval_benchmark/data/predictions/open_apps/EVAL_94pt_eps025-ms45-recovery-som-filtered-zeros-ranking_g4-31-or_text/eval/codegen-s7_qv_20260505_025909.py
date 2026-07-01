import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in the OpenApps environment.
    Q(s, a) is based on progress toward a goal, detected via accessibility tree transitions.
    """
    # 1. Detect Goal Completion
    # These keywords usually appear in the accessibility tree when a task is successfully finished.
    success_keywords = [
        'success', 'saved', 'created', 'sent', 'added', 'completed', 
        'confirmed', 'done', 'successfully', 'task finished'
    ]
    if any(kw in next_state.lower() for kw in success_keywords):
        return 1.0

    # 2. Detect Errors or Regressions
    # Error messages indicate the action was unfavorable.
    error_keywords = ['error', 'invalid', 'required', 'failed', 'incorrect', 'wrong']
    if any(kw in next_state.lower() for kw in error_keywords):
        return 0.1

    # 3. Analyze Action Intent using the State's accessibility tree
    # Extract the bid from the action (e.g., "click('12')" -> '12')
    bid_match = re.search(r"'(.*?)'", action)
    if bid_match:
        bid = bid_match.group(1)
        # Find the interactive element corresponding to the bid in the current state
        # We search for the bid identifier to see what text was associated with it.
        # Typical format in Set-of-Marks is often something like [12] or bid: 12
        bid_pattern = re.compile(rf"\[{bid}\]|bid\s*[:=]\s*{bid}")
        match = bid_pattern.search(state)
        
        if match:
            # Extract a window of text around the bid to understand what was clicked/filled
            start = max(0, match.start() - 20)
            end = min(len(state), match.end() + 40)
            context = state[start:end].lower()
            
            positive_keywords = ['save', 'submit', 'send', 'create', 'add', 'new', 'ok', 'confirm', 'yes', 'next']
            negative_keywords = ['cancel', 'delete', 'remove', 'back', 'no', 'clear']
            
            if any(kw in context for kw in positive_keywords):
                # Clicking a "Save" or "Submit" button is a high-value move toward completion.
                return 0.8
            if any(kw in context for kw in negative_keywords):
                # Clicking "Cancel" or "Delete" usually moves the agent away from the goal.
                return 0.2

    # 4. Analyze Action Primitives
    # Fill actions are generally productive as they provide necessary data for goals.
    if 'fill' in action:
        # Check if the filled text actually appears in the next state
        fill_text_match = re.findall(r"'(.*?)'", action)
        if len(fill_text_match) >= 2:
            text_filled = fill_text_match[1]
            if text_filled in next_state:
                return 0.7
        return 0.6

    # Click actions that lead to a state change are generally productive navigation.
    if 'click' in action:
        if next_state != state:
            return 0.5
        else:
            # Redundant click (state didn't change)
            return 0.2

    # Scrolling can be productive if it reveals new elements.
    if 'scroll' in action:
        return 0.4

    # No-ops provide no value.
    if 'noop' in action:
        return 0.1

    # Baseline value for undefined transitions.
    return 0.3