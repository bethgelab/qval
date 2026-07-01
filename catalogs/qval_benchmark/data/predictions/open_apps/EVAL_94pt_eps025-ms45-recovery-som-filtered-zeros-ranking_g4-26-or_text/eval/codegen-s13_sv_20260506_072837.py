import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for the given state in the OpenApps environment.
    The value represents the estimated probability of successfully achieving the 
    task goal based on the features present in the accessibility tree.
    """
    s = state.lower()
    
    # 1. Terminal State Detection (Immediate Reward)
    # High-confidence success indicators
    success_patterns = [
        r'\bsuccess\b', r'\bcompleted\b', r'\bsent\b', r'\badded\b', 
        r'\bsaved\b', r'\bconfirmed\b', r'\bdone\b', r'\btask created\b',
        r'\bmessage sent\b', r'\bevent created\b'
    ]
    for pattern in success_patterns:
        if re.search(pattern, s):
            return 0.98

    # High-confidence error indicators
    error_patterns = [
        r'\berror\b', r'\binvalid\b', r'\bfailed\b', r'\bmissing\b', 
        r'\bnot found\b', r'\bnot allowed\b'
    ]
    for pattern in error_patterns:
        if re.search(pattern, s):
            return 0.05

    # 2. Feature-based Progress Estimation
    # We use a cumulative scoring approach to approximate the probability of success.
    score = 0.1  # Base value for any valid state
    
    # A. Contextual Awareness: Are we in a task-oriented area (form, modal, etc.)?
    # Being inside a form or a dialog usually means we are closer to a terminal state.
    context_indicators = ['modal', 'dialog', 'form', 'input', 'textarea', 'popup', 'editor', 'compose', 'new ']
    if any(indicator in s for indicator in context_indicators):
        score += 0.2

    # B. Progress through Data Entry: Are form fields being filled?
    # We look for the 'value' attribute in the accessibility tree representation.
    # We cap the benefit to prevent extremely long states from artificially inflating the value.
    input_values = re.findall(r'value=["\']([^"\']+)["\']', s)
    non_empty_values = [v for v in input_values if len(v.strip()) > 0]
    score += 0.15 * min(len(non_empty_values), 4)

    # C. Action Readiness: Are there buttons or specific action keywords present?
    action_words = ['submit', 'send', 'save', 'add', 'create', 'confirm', 'apply', 'search', 'go', 'schedule']
    found_action_words = 0
    for word in action_words:
        if re.search(rf'\b{word}\b', s):
            found_action_words += 1
    
    # Add a boost for finding action-oriented keywords, capped at 4 words.
    score += 0.1 * min(found_action_words, 4)

    # D. Interaction Density: Presence of interactive roles.
    # Accessibility trees often contain role attributes for semantic identification.
    interactive_roles = re.findall(r'role=["\'](button|link|checkbox|menuitem|combobox)["\']', s)
    if len(interactive_roles) > 3:
        score += 0.1

    # 3. Normalization and Final Value Calculation
    # We cap the maximum value at 0.95 (as success is already handled) 
    # and ensure the value doesn't drop below a reasonable baseline for active states.
    final_value = min(0.95, score)
    
    # If no progress indicators (input, context, or actions) are detected, it's a low-value state.
    if score < 0.3:
        final_value = 0.1
            
    return float(round(final_value, 2))