import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment based on the 
    accessibility tree text representation.
    """
    # 1. Error Detection (Highest priority)
    # If common error/failure patterns are found, the value is likely 0.
    error_pattern = r'(?i)\b(error|invalid|failed|required|missing|incorrect|not\s+found|cannot)\b'
    if re.search(error_pattern, state):
        # Verify it is not a positive confirmation like "No errors detected"
        if not re.search(r'(?i)no\s+errors', state):
            return 0.0

    # 2. Success Detection (Highest priority)
    # If explicit success markers are present, the goal is achieved.
    success_pattern = r'(?i)\b(success|sent|saved|added|created|done|completed|confirmed|updated)\b'
    if re.search(success_pattern, state):
        return 1.0

    # 3. Progressive state estimation (Approximation)
    # We use heuristics to reward states that show task progress or high interactability.
    
    # Baseline: being in the application
    score = 0.1
    
    # A. Action-oriented elements
    # Finding words associated with finalising an action (Submit, Send, etc.)
    action_pattern = r'(?i)\b(submit|send|save|add|create|confirm|post|apply)\b'
    if re.search(action_pattern, state):
        score += 0.3
        
    # B. Form progress (Input content)
    # Accessibility trees often represent the current value of an input as a string.
    # Example: '[15] textbox "Hello World"'
    # We check for non-empty values and exclude common placeholders.
    textbox_pattern = r'(?i)(?:textbox|input|textarea)\s+"([^"]+)"'
    matches = re.findall(textbox_pattern, state)
    
    placeholders = {'search', 'type', 'enter', 'input', 'write', 'message', 'text', 'placeholder'}
    filled_count = 0
    for m in matches:
        val = m.strip()
        if val and val.lower() not in placeholders:
            filled_count += 1
    
    if filled_count > 0:
        score += 0.4
        if filled_count > 1:
            score += 0.1
    
    # C. Interaction Density
    # High numbers of interactive elements (buttons, links, etc.) often indicate 
    # an active task page rather than a splash/home page.
    interactive_pattern = r'\[\d+\]\s+(?:button|link|textbox|checkbox|radio|menuitem)'
    interactive_elements = re.findall(interactive_pattern, state, re.IGNORECASE)
    if len(interactive_elements) > 10:
        score += 0.1
    elif len(interactive_elements) > 5:
        score += 0.05

    # D. App Context Boost
    # Slight boost if the state contains keywords for the specific apps in the suite.
    app_keywords = ['todo', 'calendar', 'messenger', 'maps', 'code', 'editor']
    if any(kw in state.lower() for kw in app_keywords):
        score += 0.05

    # Cap the score at 0.95 to maintain a clear distinction from the terminal 1.0 success state.
    return min(float(score), 0.95)