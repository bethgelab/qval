import re

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an OpenApps environment by analyzing 
    the accessibility tree text representation.
    """
    # 1. Terminal State Detection
    # If the state indicates task completion, the value is 1.0.
    # We look for common success messages/phrases.
    success_patterns = [r'successfully', r'completed', r'sent', r'added', r'saved']
    for pattern in success_patterns:
        if re.search(rf'(?i)\b{pattern}\b', state):
            # We distinguish between a status message and a button label.
            # Status messages often appear as stand-alone text or are 
            # associated with high-level task completion.
            # To avoid false positives from buttons like 'Save', we check for 
            # stronger phrases if possible, but 'successfully' is almost always a message.
            if pattern == 'successfully' or pattern == 'completed':
                return 1.0

    # 2. Feature Extraction
    # Check for progress: are input fields currently filled with data?
    # Patterns like value="text" or value='text'
    filled_inputs = re.findall(r"value=['\"][^'\"]+['\"]", state)
    has_filled_input = len(filled_inputs) > 0

    # Check for readiness: are action-oriented keywords present?
    # Keywords are specific to the workflows of Todo, Calendar, Messenger, etc.
    action_keywords = ['send', 'submit', 'save', 'create', 'add', 'confirm', 'post', 'schedule', 'done', 'apply']
    has_action_keyword = False
    for kw in action_keywords:
        # Using word boundaries to avoid matching sub-strings in app names.
        if re.search(rf"(?i)\b{kw}\b", state):
            has_action_keyword = True
            break

    # Detect complexity: is this a busy dashboard or a focused task page?
    # We count interactive elements via 'bid' tags.
    interactive_elements = re.findall(r'bid=\d+', state)
    is_dashboard = len(interactive_elements) > 30

    # 3. Heuristic Scoring Logic
    # We model the value based on the typical progress through a web-based task:
    # Landing Page -> Form/Input Page -> Action Ready -> Success
    
    # Highest value: User has entered data AND the action button is visible.
    if has_filled_input and has_action_keyword:
        return 0.85
    
    # Medium value: User is in the middle of a task (e.g., typing in a form).
    elif has_filled_input:
        return 0.50
    
    # Low-Medium value: User is on a specific task page (e.g., 'Compose' view) 
    # where an action is possible, but no data is entered yet.
    elif has_action_keyword:
        if is_dashboard:
            # If it's a dashboard with many action links, the value is low.
            return 0.15
        else:
            # If it's a focused page with an action keyword, progress is likely.
            return 0.35
            
    # Lowest value: The agent is likely on a navigation page or home screen.
    else:
        return 0.05