import re

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) based on the progression towards a goal in OpenApps.
    The estimate is grounded in markers of success, the semantic value of the labels 
    of interacted elements, and the nature of the BrowserGym actions.
    """
    # 1. Direct Success Detection
    # We look for typical completion markers in the next_state that were not present in the state.
    success_markers = ['successfully', 'created', 'sent', 'completed', 'confirmed', 'saved', 'added', 'done']
    next_state_lower = next_state.lower()
    state_lower = state.lower()
    
    for marker in success_markers:
        if marker in next_state_lower and marker not in state_lower:
            return 1.0

    # 2. Action and Element Analysis
    # Extract the bid (identifier) from the action string. 
    # BrowserGym actions are formatted like click('12'), fill('5', 'text'), etc.
    bid_match = re.search(r"'(.*?)'", action)
    if not bid_match:
        # Handle actions without a specific bid (e.g., noop, scroll)
        if 'noop' in action:
            return 0.0
        if 'scroll' in action:
            return 0.1
        return 0.1

    bid = bid_match.group(1)

    # Find the label of the element associated with the bid in the state accessibility tree.
    # Accessibility trees typically look like [bid] Role "Label" ...
    # We look for the first quoted string following the bid marker.
    label_match = re.search(rf'\[{bid}\].*?"([^"]+)"', state)
    label = label_match.group(1).lower() if label_match else ""

    # 3. Semantic Label Scoring
    # High-value labels typically indicate the final step of a task.
    high_value_labels = ['save', 'submit', 'create', 'send', 'confirm', 'done', 'ok', 'finish', 'complete']
    # Mid-value labels typically indicate the start or progress of a task.
    mid_value_labels = ['add', 'new', 'edit', 'compose', 'plus', 'write', 'open', 'create']
    # Low-value labels typically indicate reversal or cancellation.
    low_value_labels = ['cancel', 'delete', 'remove', 'back', 'close', 'clear']

    # check for semantic matches in the element label
    if any(hv in label for hv in high_value_labels):
        return 0.8
    if any(mv in label for mv in mid_value_labels):
        return 0.5
    if any(lv in label for lv in low_value_labels):
        return 0.1

    # 4. Action-Type Scoring
    # If the label wasn't specifically a high/mid/low value, we score by the type of interaction.
    if 'fill' in action:
        # Filling out a form is almost always productive progress.
        return 0.4
    if 'click' in action:
        # Clicking a generic element has moderate potential.
        return 0.2
    if 'press' in action:
        # Pressing a key (like Enter) is often a confirmation step.
        return 0.3

    # Default fallback value for any other interaction.
    return 0.1